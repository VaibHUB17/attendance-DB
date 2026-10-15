import os
from contextlib import asynccontextmanager
from datetime import date as dt_date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from typing import Annotated, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field, StrictInt, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))

MIN_EPOCH_MS = 100000000000
MAX_EPOCH_MS = 4102444800000

EpochMs = Annotated[StrictInt, Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure indexes are created at startup without swallowing errors
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="idx_employees_emp_code_unique")
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="idx_employees_dept_emp_code")
    db.employees.create_index([("joined_on", ASCENDING), ("department", ASCENDING)], name="idx_employees_joined_on_dept")

    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="idx_attendance_emp_date_unique")
    db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="idx_attendance_date_desc_emp")
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="idx_attendance_emp_punch_in")
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Domain Calculations & Time Helpers
# --------------------------------------------------------------------------- #
def round_half_up(val: Optional[float | Decimal], decimals: int = 2) -> Optional[float]:
    if val is None:
        return None
    fmt = "0." + "0" * decimals
    return float(Decimal(str(val)).quantize(Decimal(fmt), rounding=ROUND_HALF_UP))


def to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    # R1: Truncate instants to whole seconds
    dt_sec = dt.replace(microsecond=0)
    if dt_sec.tzinfo is None:
        dt_sec = dt_sec.replace(tzinfo=timezone.utc)
    return int(dt_sec.timestamp() * 1000)


def from_epoch_ms(ms: int) -> datetime:
    # R1: Instants are truncated to whole seconds
    ms_truncated = (ms // 1000) * 1000
    return datetime.fromtimestamp(ms_truncated / 1000, tz=timezone.utc)


def parse_time_str(time_str: str) -> tuple[int, int]:
    h, m = map(int, time_str.split(":"))
    return h, m


def determine_attendance_date(punch_in_utc: datetime, shift_start: str, shift_end: str) -> str:
    # R1: Shift times and attendance dates are IST (UTC+05:30)
    punch_in_ist = punch_in_utc.astimezone(IST)
    is_overnight = shift_end <= shift_start

    if is_overnight:
        end_h, end_m = parse_time_str(shift_end)
        # Punch-in earlier than shift_end belongs to the shift that started previous day
        if punch_in_ist.time() < time(end_h, end_m):
            return (punch_in_ist.date() - timedelta(days=1)).isoformat()

    return punch_in_ist.date().isoformat()


def compute_late_minutes(punch_in_utc: datetime, shift_start: str, attendance_date: str) -> int:
    # R2: Grace period 10 minutes. Strictly more than 10:00 (diff_seconds > 600)
    punch_in_ist = punch_in_utc.astimezone(IST).replace(microsecond=0)
    sh_h, sh_m = parse_time_str(shift_start)
    att_date = dt_date.fromisoformat(attendance_date)
    shift_start_dt = datetime(att_date.year, att_date.month, att_date.day, sh_h, sh_m, tzinfo=IST)

    diff_seconds = (punch_in_ist - shift_start_dt).total_seconds()
    if diff_seconds > 600:
        return int(diff_seconds // 60)
    return 0


def compute_overtime(punch_out_utc: datetime, shift_start: str, shift_end: str, attendance_date: str) -> int:
    # R3: Whole minutes (floor) from shift_end to punch_out, counted only if >= 30; otherwise 0.
    punch_out_ist = punch_out_utc.astimezone(IST).replace(microsecond=0)
    end_h, end_m = parse_time_str(shift_end)
    att_date = dt_date.fromisoformat(attendance_date)

    if shift_end <= shift_start:
        shift_end_date = att_date + timedelta(days=1)
    else:
        shift_end_date = att_date

    shift_end_dt = datetime(shift_end_date.year, shift_end_date.month, shift_end_date.day, end_h, end_m, tzinfo=IST)
    diff_seconds = (punch_out_ist - shift_end_dt).total_seconds()

    if diff_seconds >= 1800:
        return int(diff_seconds // 60)
    return 0


def compute_work_hours(punch_in_utc: datetime, punch_out_utc: datetime) -> float:
    # R4: (punch_out - punch_in) in seconds / 3600, rounded to 2 decimals half-up.
    p_in = punch_in_utc.replace(microsecond=0)
    p_out = punch_out_utc.replace(microsecond=0)
    sec = (p_out - p_in).total_seconds()
    return float((Decimal(str(sec)) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_half_day(work_hours: Optional[float]) -> bool:
    # R5: Half day is true when rounded work_hours < 4.50.
    if work_hours is None:
        return False
    return work_hours < 4.50


def count_working_days(joined_on_str: str, month_str: str) -> int:
    # R7: Mon-Fri working days in the month, starting from joined_on for mid-month joiners
    year, month = map(int, month_str.split("-"))
    first_day = dt_date(year, month, 1)
    if month == 12:
        last_day = dt_date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = dt_date(year, month + 1, 1) - timedelta(days=1)

    joined_date = dt_date.fromisoformat(joined_on_str)
    start_date = max(first_day, joined_date)

    if start_date > last_day:
        return 0

    working_days = 0
    curr = start_date
    while curr <= last_day:
        if curr.weekday() < 5:  # Mon-Fri
            working_days += 1
        curr += timedelta(days=1)
    return working_days


def get_month_date_range(month_str: str) -> tuple[str, str]:
    year, month = map(int, month_str.split("-"))
    first_day = dt_date(year, month, 1)
    if month == 12:
        last_day = dt_date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = dt_date(year, month + 1, 1) - timedelta(days=1)
    return first_day.isoformat(), last_day.isoformat()


# --------------------------------------------------------------------------- #
# Serialization Helpers
# --------------------------------------------------------------------------- #
def serialize_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": to_epoch_ms(doc.get("created_at")),
    }


def serialize_history_entry(entry: dict) -> dict:
    changes_serialized = {}
    for k, v in entry.get("changes", {}).items():
        from_val = v.get("from")
        to_val = v.get("to")
        if isinstance(from_val, datetime):
            from_val = to_epoch_ms(from_val)
        if isinstance(to_val, datetime):
            to_val = to_epoch_ms(to_val)
        changes_serialized[k] = {"from": from_val, "to": to_val}

    at_val = entry.get("at")
    if isinstance(at_val, datetime):
        at_val = to_epoch_ms(at_val)

    return {
        "at": at_val,
        "by": entry.get("by"),
        "reason": entry.get("reason"),
        "changes": changes_serialized,
    }


def serialize_attendance(doc: dict) -> dict:
    history = [serialize_history_entry(h) for h in doc.get("history", [])]
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_epoch_ms(doc.get("punch_in")),
        "punch_out": to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Pydantic Schemas & Validators
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field("18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: dt_date

    @model_validator(mode="after")
    def validate_shifts(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: str = Field("PRESENT", pattern=r"^(PRESENT|WFH|ON_DUTY)$")


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeRequest(BaseModel):
    status: Optional[str] = Field(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$")
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# System & Employee Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["system"])
def health():
    try:
        db.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(status_code=503, detail="MongoDB ping failed")


@app.post("/employees", status_code=201, tags=["employees"])
def create_employee(body: EmployeeCreate):
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    doc = body.model_dump()
    doc["joined_on"] = body.joined_on.isoformat()
    doc["created_at"] = created_at

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")

    return serialize_employee(doc)


@app.get("/employees", tags=["employees"])
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = {}
    if department is not None:
        query["department"] = department

    total = db.employees.count_documents(query)
    skip = (page - 1) * page_size
    cursor = db.employees.find(query).sort("emp_code", ASCENDING).skip(skip).limit(page_size)

    items = [serialize_employee(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Attendance Endpoints
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201, tags=["attendance"])
def punch_in(body: PunchInRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        punch_in_dt = from_epoch_ms(body.punched_at)
    else:
        punch_in_dt = datetime.now(timezone.utc).replace(microsecond=0)

    att_date = determine_attendance_date(punch_in_dt, emp["shift_start"], emp["shift_end"])
    late_min = compute_late_minutes(punch_in_dt, emp["shift_start"], att_date)

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status,
        "punch_in": punch_in_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_min,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="already punched in for this date")

    return serialize_attendance(doc)


@app.post("/attendance/punch-out", status_code=200, tags=["attendance"])
def punch_out(body: PunchOutRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if body.punched_at is not None:
        punch_out_dt = from_epoch_ms(body.punched_at)
    else:
        punch_out_dt = datetime.now(timezone.utc).replace(microsecond=0)

    # Closes most recent record whose punch_in <= punched_at
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punch_out_dt}},
        sort=[("punch_in", DESCENDING)],
    )

    if not record:
        raise HTTPException(status_code=404, detail="No punch-in found")

    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="Record already punched out")

    delta_sec = (punch_out_dt - record["punch_in"]).total_seconds()
    if delta_sec <= 0 or delta_sec > 86400:
        raise HTTPException(status_code=422, detail="Invalid punch_out instant")

    work_hours = compute_work_hours(record["punch_in"], punch_out_dt)
    overtime = compute_overtime(punch_out_dt, emp["shift_start"], emp["shift_end"], record["date"])
    half_day = compute_half_day(work_hours)

    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hours,
                "overtime_minutes": overtime,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        raise HTTPException(status_code=409, detail="Record already punched out")

    return serialize_attendance(updated)


@app.get("/attendance", tags=["attendance"])
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[dt_date] = Query(None),
    date_to: Optional[dt_date] = Query(None),
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be <= date_to")

    query = {}
    if emp_code:
        query["emp_code"] = emp_code
    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from.isoformat()
        if date_to:
            query["date"]["$lte"] = date_to.isoformat()
    if status:
        query["status"] = status

    total = db.attendance_logs.count_documents(query)
    skip = (page - 1) * page_size
    cursor = db.attendance_logs.find(query).sort([("date", DESCENDING), ("emp_code", ASCENDING)]).skip(skip).limit(page_size)

    items = [serialize_attendance(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}", tags=["attendance"])
def regularize_attendance(
    emp_code: str,
    date: dt_date = Path(...),
    body: RegularizeRequest = ...,
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    date_str = date.isoformat()
    old_record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date_str})
    if not old_record:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    new_status = body.status if body.status is not None else old_record["status"]

    if body.status in ["ABSENT", "LEAVE"] and (body.punch_in is not None or body.punch_out is not None):
        raise HTTPException(status_code=422, detail="Cannot supply punch times for ABSENT or LEAVE")

    if new_status in ["ABSENT", "LEAVE"]:
        new_punch_in = None
        new_punch_out = None
    else:
        new_punch_in = from_epoch_ms(body.punch_in) if body.punch_in is not None else old_record.get("punch_in")
        new_punch_out = from_epoch_ms(body.punch_out) if body.punch_out is not None else old_record.get("punch_out")

    if new_status in ["PRESENT", "WFH", "ON_DUTY"] and new_punch_in is None:
        raise HTTPException(status_code=422, detail="Presence status requires punch_in")

    if new_punch_in is not None:
        inferred_date = determine_attendance_date(new_punch_in, emp["shift_start"], emp["shift_end"])
        if inferred_date != date_str:
            raise HTTPException(status_code=422, detail="punch_in does not belong to attendance date")

    if new_punch_out is not None:
        if new_punch_in is None:
            raise HTTPException(status_code=422, detail="punch_out requires punch_in")
        delta = (new_punch_out - new_punch_in).total_seconds()
        if delta <= 0 or delta > 86400:
            raise HTTPException(status_code=422, detail="punch_out must be after punch_in and within 24h")

    # Derived field recalculation
    if new_status in ["PRESENT", "WFH", "ON_DUTY"]:
        new_late = compute_late_minutes(new_punch_in, emp["shift_start"], date_str)
        if new_punch_out is not None:
            new_work_hours = compute_work_hours(new_punch_in, new_punch_out)
            new_overtime = compute_overtime(new_punch_out, emp["shift_start"], emp["shift_end"], date_str)
            new_half_day = compute_half_day(new_work_hours)
        else:
            new_work_hours = None
            new_overtime = 0
            new_half_day = False
    else:
        new_late = 0
        new_work_hours = None
        new_overtime = 0
        new_half_day = False

    changes = {}
    if old_record.get("status") != new_status:
        changes["status"] = {"from": old_record.get("status"), "to": new_status}

    old_in = old_record.get("punch_in")
    if (old_in != new_punch_in):
        changes["punch_in"] = {"from": old_in, "to": new_punch_in}

    old_out = old_record.get("punch_out")
    if (old_out != new_punch_out):
        changes["punch_out"] = {"from": old_out, "to": new_punch_out}

    if old_record.get("work_hours") != new_work_hours:
        changes["work_hours"] = {"from": old_record.get("work_hours"), "to": new_work_hours}

    if old_record.get("late_minutes", 0) != new_late:
        changes["late_minutes"] = {"from": old_record.get("late_minutes", 0), "to": new_late}

    if old_record.get("overtime_minutes", 0) != new_overtime:
        changes["overtime_minutes"] = {"from": old_record.get("overtime_minutes", 0), "to": new_overtime}

    if old_record.get("half_day", False) != new_half_day:
        changes["half_day"] = {"from": old_record.get("half_day", False), "to": new_half_day}

    if not changes:
        raise HTTPException(status_code=422, detail="Request changed nothing")

    correction_time = datetime.now(timezone.utc).replace(microsecond=0)
    history_item = {
        "at": correction_time,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Atomic optimistic check: verify full prior state & history length to support legacy records
    old_history_len = len(old_record.get("history") or [])
    filter_query = {
        "_id": old_record["_id"],
        "status": old_record["status"],
        "punch_in": old_record.get("punch_in"),
        "punch_out": old_record.get("punch_out"),
        "$expr": {
            "$eq": [
                {"$size": {"$ifNull": ["$history", []]}},
                old_history_len
            ]
        }
    }

    updated = db.attendance_logs.find_one_and_update(
        filter_query,
        {
            "$set": {
                "status": new_status,
                "punch_in": new_punch_in,
                "punch_out": new_punch_out,
                "work_hours": new_work_hours,
                "late_minutes": new_late,
                "overtime_minutes": new_overtime,
                "half_day": new_half_day,
            },
            "$push": {"history": history_item},
        },
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        raise HTTPException(status_code=409, detail="Concurrent regularization conflict")

    return serialize_attendance(updated)


# --------------------------------------------------------------------------- #
# Analytics Endpoints (Pure MongoDB Aggregations)
# --------------------------------------------------------------------------- #
def build_employee_monthly_pipeline(emp_code: str, month: str) -> list[dict]:
    start_d, end_d = get_month_date_range(month)
    return [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": start_d, "$lte": end_d},
            }
        },
        {
            "$addFields": {
                "parsed_date": {"$dateFromString": {"dateString": "$date"}},
            }
        },
        {
            "$addFields": {
                "day_of_week": {"$dayOfWeek": "$parsed_date"},  # 1=Sun, 2=Mon, ..., 6=Fri, 7=Sat
                "is_present_status": {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    "$is_present_status",
                                    {"$in": ["$day_of_week", [2, 3, 4, 5, 6]]},
                                ]
                            },
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"},
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly", tags=["analytics"])
def employee_monthly(
    emp_code: str,
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    working_days = count_working_days(emp["joined_on"], month)
    pipeline = build_employee_monthly_pipeline(emp_code, month)
    agg_res = list(db.attendance_logs.aggregate(pipeline))

    if agg_res:
        data = agg_res[0]
        present_days = float(data.get("present_days", 0.0))
        leave_days = int(data.get("leave_days", 0))
        late_count = int(data.get("late_count", 0))
        total_late = int(data.get("total_late_minutes", 0))
        total_ot = int(data.get("total_overtime_minutes", 0))
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late = 0
        total_ot = 0

    if working_days > 0:
        attendance_pct = round_half_up((present_days / working_days) * 100.0, 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_ot,
        "attendance_pct": attendance_pct,
    }


def build_department_summary_pipeline(month: str, department: Optional[str] = None) -> list[dict]:
    start_d, end_d = get_month_date_range(month)
    match_emp = {"joined_on": {"$lte": end_d}}
    if department:
        match_emp["department"] = department

    # Look up attendance logs per employee to avoid 16MB document limit
    return [
        {"$match": match_emp},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {
                        "$match": {
                            "date": {"$gte": start_d, "$lte": end_d}
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "date": 1,
                            "status": 1,
                            "work_hours": 1,
                            "late_minutes": 1,
                            "half_day": 1,
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$addFields": {
                "parsed_date": {
                    "$cond": [
                        {"$ifNull": ["$logs.date", False]},
                        {"$dateFromString": {"dateString": "$logs.date"}},
                        None,
                    ]
                },
                "is_present_status": {
                    "$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]
                },
            }
        },
        {
            "$addFields": {
                "row_present_days": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_present_status",
                                {"$in": [{"$dayOfWeek": "$parsed_date"}, [2, 3, 4, 5, 6]]},
                            ]
                        },
                        {"$cond": ["$logs.half_day", 0.5, 1.0]},
                        0.0,
                    ]
                },
                "row_work_hours": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_present_status",
                                {"$ne": ["$logs.work_hours", None]},
                            ]
                        },
                        "$logs.work_hours",
                        None,
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": "$department",
                "emp_set": {"$addToSet": "$emp_code"},
                "present_days": {"$sum": "$row_present_days"},
                "avg_work_hours": {"$avg": "$row_work_hours"},
                "late_count": {"$sum": {"$cond": [{"$gt": ["$logs.late_minutes", 0]}, 1, 0]}},
                "total_late_minutes": {"$sum": {"$ifNull": ["$logs.late_minutes", 0]}},
                "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
                "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}},
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$emp_set"},
                "present_days": 1,
                "avg_work_hours": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": ASCENDING}},
    ]


@app.get("/analytics/departments/summary", tags=["analytics"])
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    pipeline = build_department_summary_pipeline(month, department)
    results = list(db.employees.aggregate(pipeline))

    for item in results:
        if item.get("avg_work_hours") is not None:
            item["avg_work_hours"] = round_half_up(item["avg_work_hours"], 2)
        if item.get("present_days") is not None:
            item["present_days"] = round_half_up(item["present_days"], 2)

    return {"month": month, "items": results}


def build_late_leaderboard_pipeline(month: str, limit: int = 10, department: Optional[str] = None) -> list[dict]:
    start_d, end_d = get_month_date_range(month)
    pipeline = [
        {
            "$match": {
                "date": {"$gte": start_d, "$lte": end_d},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.extend([
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": DESCENDING},
                "output": {
                    "rank": {"$rank": {}},
                },
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": DESCENDING, "_id": ASCENDING}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ])
    return pipeline


@app.get("/analytics/leaderboard/late", tags=["analytics"])
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    pipeline = build_late_leaderboard_pipeline(month, limit, department)
    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}


def build_department_trend_pipeline(department: str, from_date: str, to_date: str, emp_codes: list[str]) -> list[dict]:
    start_dt = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc)
    next_dt = datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc) + timedelta(days=1)

    return [
        {
            "$match": {
                "emp_code": {"$in": emp_codes},
                "date": {"$gte": from_date, "$lte": to_date},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]},
                },
            }
        },
        {
            "$project": {
                "_id": 1,
                "d": {"$dateFromString": {"dateString": "$_id"}},
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$unionWith": {
                "pipeline": [
                    {
                        "$documents": [
                            {
                                "_id": from_date,
                                "d": start_dt,
                                "present_count": 0.0,
                                "late_count": 0,
                            }
                        ]
                    }
                ]
            }
        },
        {
            "$densify": {
                "field": "d",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [start_dt, next_dt],
                },
            }
        },
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$d"}},
                "d": {"$first": "$d"},
                "present_count": {"$sum": "$present_count"},
                "late_count": {"$sum": "$late_count"},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"trend_date": "$_id"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$trend_date"]},
                                ]
                            }
                        }
                    },
                    {"$count": "count"},
                ],
                "as": "hc",
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": "$_id",
                "is_working_day": {"$in": [{"$dayOfWeek": "$d"}, [2, 3, 4, 5, 6]]},
                "headcount": {"$ifNull": [{"$first": "$hc.count"}, 0]},
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_working_day",
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {"$divide": ["$present_count", "$headcount"]},
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": ASCENDING},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {"$sort": {"date": ASCENDING}},
    ]


@app.get("/analytics/departments/{department}/trend", tags=["analytics"])
def department_trend(
    department: str = Path(..., min_length=1),
    from_date: dt_date = Query(..., alias="from"),
    to_date: dt_date = Query(..., alias="to"),
):
    if to_date < from_date:
        raise HTTPException(status_code=422, detail="to must be >= from")

    if (to_date - from_date).days + 1 > 92:
        raise HTTPException(status_code=422, detail="Date range cannot exceed 92 days")

    dept_emps = list(db.employees.find({"department": department}, {"emp_code": 1}))
    if not dept_emps:
        raise HTTPException(status_code=404, detail="Department not found")

    from_str = from_date.isoformat()
    to_str = to_date.isoformat()
    emp_codes = [e["emp_code"] for e in dept_emps]

    pipeline = build_department_trend_pipeline(department, from_str, to_str, emp_codes)
    raw_items = list(db.attendance_logs.aggregate(pipeline))

    formatted_items = []
    for item in raw_items:
        rate = item.get("attendance_rate")
        m_avg = item.get("moving_avg_7d")
        formatted_items.append({
            "date": item["date"],
            "is_working_day": item["is_working_day"],
            "headcount": int(item["headcount"]),
            "present_count": float(item["present_count"]),
            "late_count": int(item["late_count"]),
            "attendance_rate": round_half_up(rate, 4) if rate is not None else None,
            "moving_avg_7d": round_half_up(m_avg, 4) if m_avg is not None else None,
        })

    return {"department": department, "items": formatted_items}


# --------------------------------------------------------------------------- #
# Admin Explain Endpoint
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}", tags=["admin"])
def explain_endpoint(
    endpoint: str = Path(..., pattern=r"^(attendance_list|employee_monthly|department_summary|late_leaderboard|department_trend)$"),
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[dt_date] = None,
    date_to: Optional[dt_date] = None,
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    from_date: Optional[dt_date] = Query(None, alias="from"),
    to_date: Optional[dt_date] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must be <= date_to")

        q = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from.isoformat()
            if date_to:
                q["date"]["$lte"] = date_to.isoformat()
        if status:
            q["status"] = status

        skip = (page - 1) * page_size
        explain_doc = db.command(
            "explain",
            {
                "find": "attendance_logs",
                "filter": q,
                "sort": {"date": -1, "emp_code": 1},
                "skip": skip,
                "limit": page_size,
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": json.loads(json_util.dumps(explain_doc)),
        }

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(status_code=422, detail="emp_code and month are required")
        pipeline = build_employee_monthly_pipeline(emp_code, month)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": json.loads(json_util.dumps(explain_doc)),
        }

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(status_code=422, detail="month is required")
        pipeline = build_department_summary_pipeline(month, department)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "employees",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": json.loads(json_util.dumps(explain_doc)),
        }

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(status_code=422, detail="month is required")
        pipeline = build_late_leaderboard_pipeline(month, limit, department)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": json.loads(json_util.dumps(explain_doc)),
        }

    elif endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(status_code=422, detail="department, from, and to are required")
        if to_date < from_date:
            raise HTTPException(status_code=422, detail="to must be >= from")
        if (to_date - from_date).days + 1 > 92:
            raise HTTPException(status_code=422, detail="Date range cannot exceed 92 days")

        dept_emps = list(db.employees.find({"department": department}, {"emp_code": 1}))
        emp_codes = [e["emp_code"] for e in dept_emps]

        pipeline = build_department_trend_pipeline(department, from_date.isoformat(), to_date.isoformat(), emp_codes)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": json.loads(json_util.dumps(explain_doc)),
        }

    raise HTTPException(status_code=422, detail=f"Unknown endpoint: {endpoint}")
