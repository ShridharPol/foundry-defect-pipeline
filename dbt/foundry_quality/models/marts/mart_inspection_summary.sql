with casting as (
    select * from {{ ref('stg_casting_metadata') }}
),

daily_summary as (
    select
        date(ingested_at) as inspection_date,
        count(*) as total_inspected,
        countif(label = 'defective') as defective_count,
        countif(label = 'ok') as ok_count,
        round(countif(label = 'defective') / count(*) * 100, 2) as defect_rate_pct
    from casting
    group by 1
),

with_rolling as (
    select
        *,
        round(avg(defect_rate_pct)
            over (order by inspection_date rows between 6 preceding and current row), 2)
            as defect_rate_7day_rolling
    from daily_summary
)

select * from with_rolling