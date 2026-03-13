with source as (
    select * from {{ source('foundry_raw', 'ai4i_maintenance') }}
),

renamed as (
    select
        udi,
        product_id,
        type,
        air_temperature_k,
        process_temperature_k,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
        machine_failure,
        twf,
        hdf,
        pwf,
        osf,
        rnf,
        ingested_at
    from source
)

select * from renamed