WITH active_users AS (
    -- Users with more than 7 sessions since 2023-01-05
    SELECT user_id
    FROM sessions
    WHERE session_start >= '2023-01-05'
    GROUP BY user_id
    HAVING COUNT(session_id) > 7
),
selected_sessions AS (
    -- Sessions of active users
    SELECT s.session_id,
        s.user_id,
        s.trip_id,
        s.session_start,
        s.session_end,
        s.flight_booked,
        s.hotel_booked,
        s.flight_discount,
        s.hotel_discount,
        s.page_clicks,
        s.cancellation
    FROM sessions s
    WHERE s.session_start >= '2023-01-05'
        AND s.user_id IN (
            SELECT user_id
            FROM active_users
        )
),
trip_values AS (
    -- Trip spend per flight+hotel
    SELECT f.trip_id,
        COALESCE(f.base_fare_usd, 0) AS flight_spend,
        COALESCE(h.nights * h.rooms * h.hotel_per_room_usd, 0) AS hotel_spend,
        COALESCE(f.base_fare_usd, 0) + COALESCE(h.nights * h.rooms * h.hotel_per_room_usd, 0) AS trip_total_usd,
        f.origin_airport,
        f.destination_airport,
        f.destination_airport_lat,
        f.destination_airport_lon
    FROM flights f
        LEFT JOIN hotels h ON f.trip_id = h.trip_id
),
user_trip_summary AS (
    -- Aggregate trips per user
    SELECT u.user_id,
        u.birthdate,
        EXTRACT(
            YEAR
            FROM AGE(CURRENT_DATE, u.birthdate)
        ) AS age,
        u.gender,
        u.married,
        u.has_children,
        SUM(tv.trip_total_usd) AS total_spent,
        COUNT(tv.trip_id) AS trips_count,
        SUM(
            CASE
                WHEN s.flight_booked = TRUE
                AND s.hotel_booked = FALSE THEN 1
                ELSE 0
            END
        ) AS flight_only,
        SUM(
            CASE
                WHEN s.flight_booked = FALSE
                AND s.hotel_booked = TRUE THEN 1
                ELSE 0
            END
        ) AS hotel_only,
        SUM(
            CASE
                WHEN s.flight_booked = TRUE
                AND s.hotel_booked = TRUE THEN 1
                ELSE 0
            END
        ) AS flight_and_hotel,
        SUM(
            CASE
                WHEN s.flight_discount = TRUE
                AND s.flight_booked = TRUE THEN 1
                ELSE 0
            END
        ) * 1.0 / NULLIF(
            SUM(
                CASE
                    WHEN s.flight_discount = TRUE THEN 1
                    ELSE 0
                END
            ),
            0
        ) AS flight_discount_conversion,
        SUM(
            CASE
                WHEN s.hotel_discount = TRUE
                AND s.hotel_booked = TRUE THEN 1
                ELSE 0
            END
        ) * 1.0 / NULLIF(
            SUM(
                CASE
                    WHEN s.hotel_discount = TRUE THEN 1
                    ELSE 0
                END
            ),
            0
        ) AS hotel_discount_conversion
    FROM users u
        LEFT JOIN selected_sessions s ON u.user_id = s.user_id
        LEFT JOIN trip_values tv ON s.trip_id = tv.trip_id
    GROUP BY u.user_id,
        u.birthdate,
        u.gender,
        u.married,
        u.has_children
) -- Final consolidated output
SELECT uts.user_id,
    uts.age,
    uts.gender,
    uts.married,
    uts.has_children,
    uts.total_spent,
    uts.trips_count,
    uts.flight_only,
    uts.hotel_only,
    uts.flight_and_hotel,
    uts.flight_discount_conversion,
    uts.hotel_discount_conversion
FROM user_trip_summary uts