drop schema if exists l67 cascade;
create schema if not exists l67;
drop role if exists operator;
drop role if exists db_user;
drop role if exists analyst;

create type l67.steering_side as enum ('LEFT', 'RIGHT');

create type l67.type_of_drive as enum ('AWD', 'FWD', 'RWD');

create type l67.body_type as enum ('Sedan', 'Wagon', 'Minibus', 'Hatchback');

create type l67.operation_type as enum ('Update', 'Insert', 'Delete');

create table if not exists l67.characteristics
(
    id                  serial primary key,
    steering_side       l67.steering_side not null,
    transmission        text              not null,
    fuel_injection_type text              not null,
    fuel_type           text              not null,
    type_of_drive       l67.type_of_drive not null,
    engine_volume       integer
);

create table if not exists l67.countries
(
    id           serial,
    country_code char(2) primary key unique,
    name         text not null
);

create table if not exists l67.brand
(
    id           serial primary key,
    name         text          not null,
    body_type    l67.body_type not null,
    country_code char(2) references l67.countries (country_code)
);

create table if not exists l67.manufacturer
(
    id   serial primary key not null,
    name text               not null
);

create table if not exists l67.car
(
    id                 serial primary key,
    name               text                                        not null,
    date_of_issue      date check ( date_of_issue < current_date ) not null,
    price              integer                                     not null,
    car_brand_id       integer references l67.brand (id),
    manufacturer_id    integer references l67.manufacturer (id),
    characteristics_id integer references l67.characteristics (id) not null
);

create table if not exists l67.showroom
(
    id      serial primary key,
    name    text not null,
    address text not null
);

create table if not exists l67.seller
(
    id          serial primary key,
    first_name  text                                 not null,
    second_name text,
    showroom_id integer references l67.showroom (id) not null
);

create table if not exists l67.selling
(
    id          serial primary key,
    car_id      integer references l67.car (id)    not null,
    seller_id   integer references l67.seller (id) not null,
    showroom_id integer references l67.showroom (id),
    date        date
);

create table if not exists l67.available_cars
(
    id          serial primary key,
    car_id      integer references l67.car (id)      not null,
    showroom_id integer references l67.showroom (id) not null,
    quantity    integer
);

-- TRIGGERS
create table l67.journal
(
    operation  l67.operation_type not null,
    stamp      timestamp          not null,
    userid     text               not null,
    table_name text               not null,
    row_id     integer            not null
);

create or replace function log() returns TRIGGER as
$$
begin

    if (tg_op = 'DELETE') then
        insert into l67.journal select 'Delete', now(), user, tg_table_name, OLD.id - 1;
        return OLD;
    elsif (tg_op = 'UPDATE') then
        insert into l67.journal select 'Update', now(), user, tg_table_name, NEW.id;
        return NEW;
    elsif (tg_op = 'INSERT') then
        insert into l67.journal select 'Insert', now(), user, tg_table_name, NEW.id;
        return NEW;
    end if;
    return null; -- result is ignored since this is an AFTER trigger
end;
$$ language 'plpgsql';

create trigger log_characteristics
    after insert or update or delete
    on l67.characteristics
    for each row
execute procedure log();

create trigger log_countries
    after insert or update or delete
    on l67.countries
    for each row
execute procedure log();

create trigger log_brand
    after insert or update or delete
    on l67.brand
    for each row
execute procedure log();

create trigger log_showroom
    after insert or update or delete
    on l67.showroom
    for each row
execute procedure log();

create trigger log_manufacturer
    after insert or update or delete
    on l67.manufacturer
    for each row
execute procedure log();

create trigger log_seller
    after insert or update or delete
    on l67.seller
    for each row
execute procedure log();

create trigger log_selling
    after insert or update or delete
    on l67.selling
    for each row
execute procedure log();

create trigger log_car
    after insert or update or delete
    on l67.car
    for each row
execute procedure log();

create trigger log_available_cars
    after insert or update or delete
    on l67.available_cars
    for each row
execute procedure log();

--=================

-- INSERTING VALUES

insert into l67.countries(country_code, name)
values ('RU', 'Russia'),
       ('JP', 'Japan'),
       ('GB', 'United Kingdom'),
       ('CN', 'China'),
       ('US', 'United States'),
       ('GE', 'Germany'),
       ('KR', 'South Korea'),
       ('UZ', 'Uzbekistan'),
       ('FR', 'France'),
       ('IT', 'Italy'),
       ('BZ', 'Brazil'),
       ('CA', 'Canada')
on conflict do nothing;

insert into l67.manufacturer(name)
values ('Toyota In Some Japanese city idk im not japanese'),
       ('BMW, Kaliningrad, Russia'),
       ('Honda, Tokyo'),
       ('Land Rover, India'),
       ('Land Rover, UK'),
       ('Jaguar, UK'),
       ('Mercedes-Benz, Novgorod'),
       ('Lada, Tolyatti'),
       ('Lamborghini, Italy'),
       ('Hyundai, Russia'),
       ('Hyundai, Korea'),
       ('Hyundai, US'),
       ('Kia, US'),
       ('Kia, Korea'),
       ('Citroen, France'),
       ('Daewoo, Uzbekistan'),
       ('Volkswagen, Brazil'),
       ('Volkswagen, Germany'),
       ('Volkswagen, Russia'),
       ('Ford, US'),
       ('Ford, India'),
       ('Haval, China'),
       ('Haval, Russia'),
       ('Chevrolet, US'),
       ('AVTO, China'),
       ('CAR, Canada'),
       ('AVTO, Canada')
on conflict do nothing;

insert into l67.brand(name, country_code, body_type)
values ('Toyota', 'JP', 'Sedan'),
       ('BMW', 'GE', 'Sedan'),
       ('Honda', 'JP', 'Wagon'),
       ('Land Rover', 'GB', 'Minibus'),
       ('Jaguar', 'GB', 'Sedan'),
       ('Mercedes-Benz', 'GE', 'Sedan'),
       ('Lada', 'RU', 'Sedan'),
       ('Lamborghini', 'IT', 'Minibus'),
       ('Hyundai', 'KR', 'Minibus'),
       ('Kia', 'KR', 'Sedan'),
       ('Citroen', 'FR', 'Sedan'),
       ('Daewoo', 'UZ', 'Sedan'),
       ('Volkswagen', 'GE', 'Hatchback'),
       ('Ford', 'US', 'Minibus'),
       ('Haval', 'CN', 'Minibus'),
       ('Lada', 'RU', 'Sedan'),
       ('Chevrolet', 'US', 'Sedan'),
       ('Toyota', 'JP', 'Wagon'),
       ('CAR', 'CA', 'Wagon'),
       ('AVTO', 'CN', 'Wagon'),
       ('AVTO', 'CA', 'Minibus')
on conflict do nothing;

insert into l67.characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive,
                                engine_volume)
values ('RIGHT', 'Automatic', 'TSI', 'Gasoline', 'FWD', 2200),
       ('LEFT', 'Manual', 'KE-Jetronic', 'Gasoline', 'RWD', 2800),
       ('LEFT', 'CVT', 'MPI', 'Gasoline', 'FWD', 3000),
       ('LEFT', 'Automatic', 'TSI', 'Diesel', 'FWD', 3000),
       ('LEFT', 'Robotic', 'D4', 'Gasoline', 'FWD', 4800),
       ('LEFT', 'Manual', 'MPI', 'Gasoline', 'AWD', 1600),
       ('LEFT', 'Automatic', 'None', 'Electric', 'RWD', null),
       ('LEFT', 'CVT', 'TSI', 'Gasoline', 'FWD', '2500'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'FWD', null),
       ('LEFT', 'Robotic', 'TSI', 'Diesel', 'AWD', 2000),
       ('RIGHT', 'Automatic', 'MPI', 'Gasoline', 'FWD', 1100)
on conflict do nothing;

insert into l67.car(name, manufacturer_id, date_of_issue, price, car_brand_id, characteristics_id)
values ('Vitz', 1, '2002-12-11', 400000, 1, 1),
       ('520D', 2, '2021-11-19', 4000000, 2, 4),
       ('Accord', 3, '2008-10-01', 800000, 3, 3),
       ('Discovery 3', 4, '2005-06-05', 1000000, 4, 4),
       ('Discovery 3', 5, '2006-09-13', 1100000, 4, 4),
       ('XF', 6, '2011-05-16', 800000, 5, 1),
       ('Vista', 1, '1994-06-11', 215000, 1, 1),
       ('Corolla', 1, '1994-06-11', 480000, 18, 1),
       ('Corolla', 1, '1994-06-11', 480000, 1, 11),
       ('Corolla', 1, '1994-06-11', 480000, 1, 1),
       ('Sprinter', 1, '1994-06-11', 140000, 18, 11),
       ('S500', 7, '2010-09-13', 2000000, 6, 2),
       ('Vesta', 8, '2018-11-06', 1300000, 7, 6),
       ('Urus', 9, '2022-02-17', 40000000, 8, 10),
       ('Solaris', 10, '2022-04-11', 1600000, 9, 6),
       ('Creta', 11, '2022-02-24', 2100000, 9, 6),
       ('Rio', 12, '2021-03-03', 1600000, 10, 6),
       ('Mohave', 13, '2022-01-06', 4000000, 10, 8),
       ('C4', 15, '2013-09-11', 800000, 11, 2),
       ('C4', 15, '2013-09-11', 800000, 11, 2),
       ('Nexia', 16, '2011-01-05', 200000, 12, 6),
       ('Polo', 18, '2022-04-16', 1400000, 13, 6),
       ('Mondeo', 20, '2010-01-24', 400000, 14, 6),
       ('F7', 22, '2022-01-07', 2100000, 15, 8),
       ('Granta', 8, '2019-01-10', 800000, 7, 6),
       ('Camaro', 24, '2021-10-12', 2500000, 17, 2),
       ('Corolla', 1, '2020-06-11', 500000, 1, 11),
       ('Camry', 1, '2022-06-11', 1600000, 1, 6),
       ('Land Cruiser', 1, '2020-06-11', 4800000, 1, 4),
       ('C4', 15, '2022-09-11', 4000000, 11, 2),
       ('Odyssey', 3, '2018-01-01', 950000, 3, 3),
       ('Accord', 3, '2018-01-01', 950000, 3, 3),
       ('Stream', 3, '2018-01-01', 950000, 3, 3),
       ('Wizard', 25, '2022-06-11', 1600000, 20, 6),
       ('Fleet', 26, '2022-06-11', 1600000, 19, 6),
       ('Screamer', 27, '2020-06-11', 1800000, 21, 6)
on conflict do nothing;

insert into l67.showroom(name, address)
values ('АЦ Фрунзе', 'Фрунзе, 252'),
       ('Восток моторс', 'Большевистская, 276/2'),
       ('АЦ Сибирский тракт', 'Жуковского, 96/2'),
       ('Эксперт НСК', 'Большевистская, 276/1')
on conflict do nothing;

insert into l67.seller(first_name, second_name, showroom_id)
values ('Viktor', 'Vatov', 1),
       ('Dmitry', 'Pelevin', 2),
       ('Ilya', 'Amogusov', 3),
       ('Ivan', 'Kalita', 4)
on conflict do nothing;

insert into l67.available_cars(car_id, showroom_id, quantity)
values (1, 1, 1),
       (2, 1, 1),
       (3, 1, 1),
       (4, 2, 1),
       (5, 2, 1),
       (6, 2, 1),
       (7, 3, 1),
       (8, 3, 1),
       (9, 3, 1),
       (10, 4, 1),
       (11, 4, 1),
       (12, 4, 1),
       (28, 2, 1),
       (29, 2, 1),
       (30, 3, 1),
       (31, 2, 1),
       (32, 2, 1),
       (33, 2, 1),
       (34, 1, 1),
       (35, 4, 1),
       (36, 1, 1)
on conflict do nothing;

insert into l67.selling(car_id, seller_id, showroom_id, date)
values (13, 1, 1, '2020-10-11'),
       (14, 2, 2, '2019-09-01'),
       (15, 3, 3, '2018-08-07'),
       (16, 4, 4, '2017-07-06'),
       (17, 2, 1, '2022-06-12'),
       (18, 3, 2, '2011-05-14'),
       (19, 1, 3, '2015-04-21'),
       (20, 4, 4, '2016-03-22'),
       (21, 1, 1, '2021-02-24'),
       (22, 2, 2, '2020-01-16'),
       (23, 3, 4, '2018-12-18'),
       (24, 1, 3, '2016-11-05'),
       (25, 4, 2, '2020-09-14'),
       (26, 2, 1, '2021-10-18'),
       (27, 4, 3, '2022-01-19')
on conflict do nothing;

create index countries_idx on l67.countries using hash (name);
create index available_cars_idx on l67.car using hash (name);

-- Changing roles

create role operator with login password 'qwerty';
create role db_user with login password 'qwerty';
create role analyst with login;

revoke all privileges on all tables in schema l67 from operator;
revoke all privileges on all tables in schema l67 from db_user;
revoke all privileges on all tables in schema l67 from analyst;

grant insert, select, update, delete on l67.manufacturer, l67.countries, l67.showroom, l67.brand, l67.characteristics to operator;

grant insert, select, update, delete on l67.car, l67.characteristics, l67.seller, l67.selling, l67.available_cars to db_user;

grant select on all tables in schema l67 to analyst;
revoke all privileges on l67.journal from analyst;

-- FUNCTIONS

create or replace function l67.add_auto(_car_id int, _showroom_id int, _count integer) returns char(30)
as
$$
declare
    count int;
begin
    if (select count(id)
        from l67.car
        where id = _car_id)
        = 0 then
        return 'Car not found';
    end if;

    if (select count(id) from l67.showroom where id = _showroom_id)
        = 0 then
        return 'Showroom not found';
    end if;

    select count(cars.car_id)
    into count
    from l67.available_cars as cars
    where _showroom_id = cars.showroom_id;

    if count = 0 then
        insert into l67.available_cars (car_id, showroom_id)
        values (_car_id, _showroom_id);
    else
        update
            l67.available_cars as cars
        set quantity = cars.quantity + _count
        where cars.car_id = _car_id
          and cars.showroom_id = _showroom_id;
    end if;

    return 'OK';
end;
$$
    language 'plpgsql';

--============

create or replace function l67.sell_auto(_car_id int, _seller_id int) returns char(30)
as
$$
declare
    _showroom_id  int;
    declare count int;
begin
    if (select count(id)
        from l67.car
        where id = _car_id)
        = 0 then
        return 'Car model not found';
    end if;

    if (select count(id)
        from l67.seller
        where id = _seller_id)
        = 0 then
        return 'Seller not found';
    end if;

    select showroom_id
    into _showroom_id
    from l67.seller
    where id = _seller_id;

    select count(car_id)
    into count
    from l67.available_cars as cars
    where cars.showroom_id = _showroom_id
      and cars.car_id = _car_id;

    if count = 0 then
        return 'There is no cars that model available';
    else
        if (select quantity
            from l67.available_cars as cars
            where cars.car_id = _car_id
              and cars.showroom_id = _showroom_id) = 1
        then
            delete
            from l67.available_cars as cars
            where cars.car_id = _car_id
              and cars.showroom_id = _showroom_id;
        else
            update l67.available_cars as cars
            set quantity = cars.quantity - 1
            where cars.car_id = _car_id
              and cars.showroom_id = _showroom_id;
        end if;
        insert into l67.selling(car_id, seller_id, showroom_id, date)
        values (_car_id, _seller_id, _showroom_id, now());
    end if;
    return 'OK';
end;
$$
    language 'plpgsql';

--============

create or replace function l67.hire_seller(_first_name text, _second_name text, _showroom_id int) returns char(20) as
$$
begin
    if (select count(id) from l67.showroom where id = _showroom_id) = 0 then
        return 'Showroom not found';
    end if;
    insert into l67.seller(first_name, second_name, showroom_id) values (_first_name, _second_name, _showroom_id);
    return 'OK';
end;
$$
    language 'plpgsql';

--============

create or replace function l67.fire_seller(seller_id int)
    returns char(20) as
$$
begin
    if (select count(id) from l67.seller where id = seller_id) = 0 then
        return 'Seller not found';
    end if;
    delete from l67.seller where id = seller_id;
    return 'OK';
end;
$$
    language 'plpgsql';

--============

create or replace function l67.get_showroom_rating(_showroom_id int)
    returns table
            (
                name   text,
                Rating integer
            )
as
$$
begin
    if (select count(id) from l67.showroom where id = _showroom_id)
        = 0
    then
        return;
    end if;
    return query select l67.car.name::text,
                        rating.Rating::integer as rating
                 from (select l67.selling.car_id,
                              count(l67.selling.car_id) as Rating
                       from l67.selling
                       where showroom_id = _showroom_id
                       group by l67.selling.car_id) as rating
                          inner join
                      l67.car on rating.car_id = car.id
                 order by rating desc;
end;
$$
    language 'plpgsql';

--============

create or replace function l67.get_seller_rating(_seller_id integer)
    returns table
            (
                car_model text,
                rating    integer
            )
as
$$
begin
    return query select l67.car.name::text,
                        rating.Rating::integer as rating
                 from (select l67.selling.car_id,
                              count(l67.selling.car_id) as Rating
                       from l67.selling
                       where seller_id = _seller_id
                       group by l67.selling.car_id) as rating
                          inner join
                      l67.car on rating.car_id = car.id
                 order by rating desc;
end;
$$
    language 'plpgsql';

create or replace function l67.get_country_sales_volume(_country_code text)
    returns table
            (
                car_brand    text,
                sales_volume integer
            )
as
$$
begin
    return query select sales.name::text,
                        sales.volume::integer as sales_volume
                 from (select brand.name                as name,
                              count(l67.selling.car_id) as volume
                       from l67.selling
                                inner join l67.car as car on selling.car_id = car.id
                                inner join l67.brand as brand on car.car_brand_id = brand.id
                       where brand.country_code = _country_code
                       group by brand.name) as sales

                 order by sales_volume desc;
end;
$$
    language 'plpgsql';

create or replace function l67.get_prices_for_country(_country_code text)
    returns table
            (
                min_price text,
                avg_price text,
                max_price text
            )
as
$$
begin
    return query select (sales.min_price::integer || ' RUB') as min_price,
                        (sales.avg_price::integer || ' RUB') as avg_price,
                        (sales.max_price::integer || ' RUB') as max_price
                 from (select min(car.price) as min_price,
                              avg(car.price) as avg_price,
                              max(car.price) as max_price
                       from l67.available_cars
                                inner join l67.car as car on available_cars.car_id = car.id
                                inner join l67.brand as brand on car.car_brand_id = brand.id
                       where brand.country_code = _country_code) as sales
                 order by max_price desc;
end;
$$
    language 'plpgsql';

create index if not exists available_idx on l67.available_cars (showroom_id, car_id);
create index if not exists selling_idx on l67.selling (showroom_id, car_id, seller_id);
create index if not exists brand_idx on l67.brand (country_code);
select * from l67.get_prices_for_country('JP');
select * from l67.get_prices_for_country('GE');