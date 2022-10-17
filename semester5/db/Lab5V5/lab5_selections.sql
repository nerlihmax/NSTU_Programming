-- cars assembled in country
select car.id,
       brand.name as brand,
       car.name   as model,
       c.name     as country
from l5.car as car
         inner join l5.brand brand on brand.id = car.car_brand_id
         inner join l5.countries c on c.country_code = brand.country_code
where c.name = 'Japan';

-- cars <1.3L and price <600_000
select car.id,
       brand.name       as brand,
       car.name         as model,
       ch.engine_volume as engine_volume,
       car.price        as price
from l5.car as car
         inner join l5.brand brand on brand.id = car.car_brand_id
         inner join l5.characteristics as ch on car.characteristics_id = ch.id
where ch.engine_volume < 1300
  and car.price < 600000;

-- japanese cars AWD < 6 month old
select car.id,
       brand.name   as brand,
       car.name     as model,
       ch.type_of_drive,
       car.date_of_issue,
       country.name as country
from l5.car as car
         inner join l5.brand as brand on car.car_brand_id = brand.id
         inner join l5.characteristics ch on car.characteristics_id = ch.id
         inner join l5.countries country on country.country_code = brand.country_code
where country.name = 'Japan'
  and ch.type_of_drive = 'AWD'
  and car.date_of_issue > current_date::date - interval '6 months';

-- wagon not from russia with left steering wheel side
select car.id,
       brand.name      as brand,
       car.name        as model,
       ch.steering_side,
       country.name,
       brand.body_type as body_type
from l5.car
         inner join l5.brand on brand.id = car.car_brand_id
         inner join l5.characteristics as ch on car.characteristics_id = ch.id
         inner join l5.countries as country on brand.country_code = country.country_code
where ch.steering_side = 'LEFT'
  and brand.body_type = 'Wagon'
  and country.name != 'Russia';

-- wagons and minibuses from china, canada
select car.id,
       brand.name      as brand,
       car.name        as model,
       country.name,
       brand.body_type as body_type
from l5.car
         inner join l5.brand on brand.id = car.car_brand_id
         inner join l5.countries as country on brand.country_code = country.country_code
where (brand.body_type = 'Wagon' or brand.body_type = 'Minibus')
  and (country.name = 'China' or country.name = 'Canada')
  and (brand.name = 'AVTO' or brand.name = 'CAR');
