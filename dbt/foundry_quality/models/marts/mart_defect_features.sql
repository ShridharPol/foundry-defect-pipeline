with casting as (
    select * from {{ ref('stg_casting_metadata') }}
),

secom as (
    select * from {{ ref('stg_secom') }}
),

casting_features as (
    select
        filename,
        label,
        split,
        gcs_uri,
        case when label = 'defective' then 1 else 0 end as is_defective
    from casting
),

secom_summary as (
    select
        avg(feature_0) as avg_feature_0,
        avg(feature_1) as avg_feature_1,
        avg(feature_2) as avg_feature_2,
        countif(label = 1) as total_fails,
        countif(label = -1) as total_passes,
        round(countif(label = 1) / count(*) * 100, 2) as fail_rate_pct
    from secom
)

select
    c.*,
    s.fail_rate_pct as secom_fail_rate,
    s.total_fails as secom_total_fails
from casting_features c
cross join secom_summary s