--selecting sessions based on Elenas suggestion. 
------------------
with active_users as (
    select user_id,
        count(session_id) as total_session
    from sessions
    where session_start >= '2023-01-05'
    group by user_id
    having count(session_id) > 7
)
select *
from sessions
where sessions.session_start >= '2023-01-05'
    and sessions.user_id in (
        select user_id
        from active_users
    );