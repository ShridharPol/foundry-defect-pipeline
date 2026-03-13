with ai4i as (
    select * from {{ ref('stg_ai4i') }}
),

process_summary as (
    select
        type as machine_type,
        count(*) as total_records,
        countif(machine_failure = 1) as failure_count,
        round(countif(machine_failure = 1) / count(*) * 100, 2) as failure_rate_pct,
        sum(twf) as tool_wear_failures,
        sum(hdf) as heat_dissipation_failures,
        sum(pwf) as power_failures,
        sum(osf) as overstrain_failures,
        sum(rnf) as random_failures,
        round(avg(air_temperature_k), 2) as avg_air_temp_k,
        round(avg(process_temperature_k), 2) as avg_process_temp_k,
        round(avg(rotational_speed_rpm), 2) as avg_rotational_speed,
        round(avg(torque_nm), 2) as avg_torque_nm,
        round(avg(tool_wear_min), 2) as avg_tool_wear_min
    from ai4i
    group by 1
)

select * from process_summary