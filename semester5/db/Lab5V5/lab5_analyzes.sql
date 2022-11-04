create or replace function add_n(n integer) returns char
as
$$
declare
    t int;
begin
    select max(id) into t from l5.brand;
    for k in (t + 1)..(n + t + 1)
        loop
            insert into l5.brand(name, body_type, country_code)
            values ((select (array ['Toyota', 'Ford','BMW'])[floor(random() * 3 + 1)]),
                    (select (array ['Sedan', 'Minibus','Hatchback'])[floor(random() * 3 + 1)])::l5.body_type,
                    (select (array ['US', 'JP','GE'])[floor(random() * 3 + 1)]));
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$
    language 'plpgsql';

select add_n(1000);

-- analyse1
explain analyse
select *
from l5.brand
where country_code = 'US';

-- analyse2
explain analyse
select *
from l5.brand
where country_code = 'JP';

-- analyse3
explain analyse
select *
from l5.brand
where country_code = 'GE';

create index if not exists brand_country_code_idx on l5.brand (country_code);

-- analyse1_btree
explain analyse
select *
from l5.brand
where country_code = 'US';

-- analyse2_btree
explain analyse
select *
from l5.brand
where country_code = 'JP';

-- analyse3_btree
explain analyse
select *
from l5.brand
where country_code = 'GE';

drop index if exists brand_country_code_idx;

create index if not exists brand_country_code_idx on l5.brand using hash (country_code);

-- analyse1_hash
explain analyse
select *
from l5.brand
where country_code = 'US';

-- analyse2_hash
explain analyse
select *
from l5.brand
where country_code = 'JP';

-- analyse3_hash
explain analyse
select *
from l5.brand
where country_code = 'GE';

drop index if exists brand_country_code_idx;

-- analyse1_lower
explain analyse
select *
from l5.brand
where lower(country_code) = 'us';

-- analyse2_lower
explain analyse
select *
from l5.brand
where lower(country_code) = 'jp';

-- analyse3_lower
explain analyse
select *
from l5.brand
where lower(country_code) = 'ge';

create index brand_country_code_idx on l5.brand (lower(country_code));

-- analyse1_lower_idx
explain analyse
select *
from l5.brand
where lower(country_code) = 'us';

-- analyse2_lower_idx
explain analyse
select *
from l5.brand
where lower(country_code) = 'jp';

-- analyse3_lower_idx
explain analyse
select *
from l5.brand
where lower(country_code) = 'ge';