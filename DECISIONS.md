# DECISIONS.md (about 150-200 words in total - plain language, your own words)

1. **Indexes.**
   `employees`: unique `{"emp_code": 1}`; `{"department": 1, "emp_code": 1}`; `{"joined_on": 1, "department": 1}`.
   `attendance_logs`: unique `{"emp_code": 1, "date": 1}`; `{"date": -1, "emp_code": 1}`; `{"emp_code": 1, "punch_in": -1}`.
   Rejected: `(emp_code, punch_out, punch_in)` (range filters broke punch-in sort) and standalone `late_minutes` (queries filter by date first).

2. **Punch-in race.**
   Concurrent requests race on `insert_one`. The unique index on `(emp_code, date)` guarantees atomic serialization: the first insert commits (201), while the second raises `DuplicateKeyError`, handled as 409.

3. **Ties.**
   The pipeline ranks employees using `$setWindowFields` with `$rank` (1, 2, 2, 4). Because filtering (`rank <= limit`) occurs after ranking, both tied employees at the cutoff share the rank and are returned, ordered by `emp_code` ascending.

4. **Headcount.**
   The pipeline starts from `employees` (`joined_on <= month_end`), lookups logs per employee, unwinds with `preserveNullAndEmptyArrays: true`, and groups by department using `$addToSet` on `emp_code`. Zero-log employees are preserved in headcount without hitting 16 MB BSON limits.

5. **One thing you would change for 100x data.**
   Pre-aggregate daily metrics and monthly totals into dedicated rollup collections via Change Streams or scheduled triggers, converting multi-collection aggregations into indexed point reads.
