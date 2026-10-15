import os
from contextlib import asynccontextmanager
from datetime import date as dt_date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field, StrictInt, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))

MIN_EPOCH_MS = 100000000000
MAX_EPOCH_MS = 9999999999999
EpochMs = Annotated[StrictInt, Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Idempotently create required indexes on startup
    db.employees.create_index([("emp_code", ASCENDING)], unique=True)
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)])
    db.employees.create_index([("joined_on", ASCENDING), ("department", ASCENDING)])

    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True
    )
    db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)])
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)])
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers & Business Logic (R1 - R5)
# --------------------------------------------------------------------------- #
def round_half_up(val: float | Decimal, places: int = 2) -> float:
    d = Decimal(str(val))
    target = Decimal("10") ** -places
    return float(d.quantize(target, rounding=ROUND_HALF_UP))


def to_epoch_ms(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def from_epoch_ms(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def parse_time_str(t_str: str) -> time:
    h, m = map(int, t_str.split(":"))
    return time(hour=h, minute=m)


def determine_attendance_date(punch_in_utc: datetime, shift_start_str: str, shift_end_str: str) -> str:
    punch_in_ist = punch_in_utc.astimezone(IST)
    start_t = parse_time_str(shift_start_str)
    end_t = parse_time_str(shift_end_str)

    if end_t <= start_t:
        # Overnight shift: if punched in before shift_end on the calendar day, belongs to previous date
        if punch_in_ist.time() < end_t:
            return (punch_in_ist.date() - timedelta(days=1)).isoformat()
    return punch_in_ist.date().isoformat()


def compute_late_minutes(punch_in_utc: datetime, shift_start_str: str, attendance_date_str: str) -> int:
    d = dt_date.fromisoformat(attendance_date_str)
    start_t = parse_time_str(shift_start_str)
    shift_start_ist = datetime.combine(d, start_t, tzinfo=IST)

    punch_in_ist = punch_in_utc.astimezone(IST)
    diff_seconds = (punch_in_ist - shift_start_ist).total_seconds()

    if diff_seconds > 600:
        return int(diff_seconds // 60)
    return 0


def compute_work_hours(punch_in_utc: datetime, punch_out_utc: datetime) -> float:
    diff_sec = (punch_out_utc - punch_in_utc).total_seconds()
    raw_hours = Decimal(diff_sec) / Decimal(3600)
    return round_half_up(raw_hours, 2)


def compute_overtime(punch_out_utc: datetime, shift_start_str: str, shift_end_str: str, attendance_date_str: str) -> int:
    d = dt_date.fromisoformat(attendance_date_str)
    start_t = parse_time_str(shift_start_str)
    end_t = parse_time_str(shift_end_str)

    if end_t <= start_t:
        shift_end_date = d + timedelta(days=1)
    else:
        shift_end_date = d

    shift_end_ist = datetime.combine(shift_end_date, end_t, tzinfo=IST)
    punch_out_ist = punch_out_utc.astimezone(IST)

    diff_seconds = (punch_out_ist - shift_end_ist).total_seconds()
    if diff_seconds >= 1800:
        return int(diff_seconds // 60)
    return 0


def compute_half_day(work_hours: float) -> bool:
    return work_hours < 4.0


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


def serialize_attendance(doc: dict) -> dict:
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
        "history": [
            {
                "at": to_epoch_ms(h.get("at")),
                "by": h.get("by"),
                "reason": h.get("reason"),
                "changes": {
                    k: (to_epoch_ms(v) if k in ["punch_in", "punch_out"] and isinstance(v, datetime) else v)
                    for k, v in h.get("changes", {}).items()
                },
            }
            for h in doc.get("history", [])
        ],
    }


# --------------------------------------------------------------------------- #
# Pydantic Schemas
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1)
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1)
    shift_start: str = Field("09:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field("18:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    joined_on: dt_date

    @model_validator(mode="after")
    def validate_shift_times(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


class PunchInRequest(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    punched_at: Optional[EpochMs] = None
    status: str = Field("PRESENT", pattern=r"^(PRESENT|WFH|ON_DUTY)$")


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
    department: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = {}
    if department is not None:
        query["department"] = department

    total = db.employees.count_documents(query)
    skip = (page - 1) * page_size
    cursor = (
        db.employees.find(query)
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )
    items = [serialize_employee(d) for d in cursor]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# --------------------------------------------------------------------------- #
# Attendance Endpoints (Part A)
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201, tags=["attendance"])
def punch_in(body: PunchInRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    punch_dt = from_epoch_ms(body.punched_at) if body.punched_at is not None else datetime.now(timezone.utc).replace(microsecond=0)
    attendance_date = determine_attendance_date(punch_dt, emp["shift_start"], emp["shift_end"])

    late_min = compute_late_minutes(punch_dt, emp["shift_start"], attendance_date)

    doc = {
        "emp_code": body.emp_code,
        "date": attendance_date,
        "status": body.status,
        "punch_in": punch_dt,
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


@app.get("/attendance", tags=["attendance"])
def list_attendance(
    emp_code: Optional[str] = Query(None, pattern=r"^EMP\d{4,6}$"),
    date_from: Optional[dt_date] = Query(None),
    date_to: Optional[dt_date] = Query(None),
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|HALF_DAY|WFH|ON_DUTY|LEAVE)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from cannot be after date_to")

    query = {}
    if emp_code:
        query["emp_code"] = emp_code
    if status:
        query["status"] = status

    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from.isoformat()
        if date_to:
            query["date"]["$lte"] = date_to.isoformat()

    total = db.attendance_logs.count_documents(query)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(query)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )

    items = [serialize_attendance(d) for d in cursor]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# --------------------------------------------------------------------------- #
# TODO - the rest of the contract (see openapi.yaml):
#   POST  /attendance/punch-out
#   PATCH /attendance/{emp_code}/{date}
#   GET   /analytics/employees/{emp_code}/monthly
#   GET   /analytics/departments/summary
#   GET   /analytics/leaderboard/late
#   GET   /analytics/departments/{department}/trend
#   GET   /admin/explain/{endpoint}
# --------------------------------------------------------------------------- #
