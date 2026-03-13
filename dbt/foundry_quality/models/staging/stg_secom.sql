with source as (
    select * from {{ source('foundry_raw', 'secom_sensors') }}
),

renamed as (
    select
        * except(ingested_at),
        ingested_at
    from source
)

select * from renamed