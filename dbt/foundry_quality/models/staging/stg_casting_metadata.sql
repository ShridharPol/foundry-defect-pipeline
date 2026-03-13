with source as (
    select * from {{ source('foundry_raw', 'casting_metadata') }}
),

renamed as (
    select
        filename,
        label,
        split,
        gcs_uri,
        ingested_at
    from source
)

select * from renamed