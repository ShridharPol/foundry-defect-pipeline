with casting as (
    select * from {{ ref('stg_casting_metadata') }}
),

with_shift as (
    select *,
        case
            when extract(hour from timestamp(ingested_at)) between 6 and 13 then 'morning'
            when extract(hour from timestamp(ingested_at)) between 14 and 21 then 'afternoon'
            else 'night'
        end as shift
    from casting
),

shift_summary as (
    select
        shift,
        count(*) as total_inspected,
        countif(label = 'defective') as defective_count,
        round(countif(label = 'defective') / count(*) * 100, 2) as defect_rate_pct
    from with_shift
    group by 1
)

select * from shift_summary