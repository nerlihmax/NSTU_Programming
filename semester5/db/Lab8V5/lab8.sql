drop schema if exists l8 cascade;
create schema if not exists l8;

create type l8.steering_side as enum ('LEFT', 'RIGHT');

create type l8.type_of_drive as enum ('AWD', 'FWD', 'RWD');

create type l8.body_type as enum ('Sedan', 'Wagon', 'Minibus', 'Hatchback');

create type l8.operation_type as enum ('Update', 'Insert', 'Delete');

create table if not exists l8.characteristics
(
    id                  serial primary key,
    steering_side       l8.steering_side not null,
    transmission        text             not null,
    fuel_injection_type text             not null,
    fuel_type           text             not null,
    type_of_drive       l8.type_of_drive not null
);

create table if not exists l8.countries
(
    id           serial,
    country_code char(2) primary key unique,
    name         text not null
);

create table if not exists l8.brand
(
    id            serial primary key,
    name          text         not null,
    city          text,
    body_type     l8.body_type not null,
    engine_volume integer,
    country_code  char(2) references l8.countries (country_code)
);

create table if not exists l8.manufacturer
(
    id   serial primary key not null,
    name text               not null
);

create table if not exists l8.car
(
    id                 serial primary key,
    name               text                                        not null,
    date_of_issue      date check ( date_of_issue < current_date ) not null,
    price              integer                                     not null,
    car_brand_id       integer references l8.brand (id),
    manufacturer_id    integer references l8.manufacturer (id),
    characteristics_id integer references l8.characteristics (id)  not null
);

create table if not exists l8.showroom
(
    id      serial primary key,
    name    text not null,
    address text not null
);

create table if not exists l8.seller
(
    id          serial primary key,
    first_name  text                                not null,
    second_name text,
    showroom_id integer references l8.showroom (id) not null
);

create table if not exists l8.selling
(
    id          serial primary key,
    car_id      integer references l8.car (id)    not null,
    seller_id   integer references l8.seller (id) not null,
    showroom_id integer references l8.showroom (id),
    date        date
);

create table if not exists l8.available_cars
(
    id          serial primary key,
    car_id      integer references l8.car (id)      not null,
    showroom_id integer references l8.showroom (id) not null,
    quantity    integer
);

-- TRIGGERS

create or replace function l8.delete_brand() returns TRIGGER as
$$
begin
    delete from l8.brand where id = old.id;
    delete from l8.car where car_brand_id = old.id;
    return old;
end;
$$ language 'plpgsql';

create trigger delete_brand_cascade
    after delete
    on l8.brand
    for each row
execute procedure l8.delete_brand();

create or replace function l8.add_brand() returns trigger as
$$
begin
    if new.name is null then
        new.name = 'Manufacturer|' + round(random() * 100000 + 1)::text;
    end if;

    if new.city is null then
        new.city = 'Tokyo';
    end if;

    if new.engine_volume is null then
        if new.city = 'Tokyo'
        then
            new.engine_volume = 1300;
        else
            new.engine_volume = 0;
        end if;
    end if;
    return new;
end;
$$ language 'plpgsql';

create trigger add_brand_tg
    before insert
    on l8.brand
    for each row
execute procedure l8.add_brand();

-- INSERTING VALUES

insert into l8.countries(country_code, name)
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

insert into l8.manufacturer(name)
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

insert into l8.brand(name, country_code, body_type, engine_volume, city)
values ('Toyota', 'JP', 'Sedan', null, null),
       ('BMW', 'GE', 'Sedan', null, null),
       ('Honda', 'JP', 'Wagon', null, null),
       ('Land Rover', 'GB', 'Minibus', null, null),
       ('Jaguar', 'GB', 'Sedan', null, null),
       ('Mercedes-Benz', 'GE', 'Sedan', null, null),
       ('Lada', 'RU', 'Sedan', null, null),
       ('Lamborghini', 'IT', 'Minibus', null, null),
       ('Hyundai', 'KR', 'Minibus', null, 'Seoul'),
       ('Kia', 'KR', 'Sedan', null, null),
       ('Citroen', 'FR', 'Sedan', null, null),
       ('Daewoo', 'UZ', 'Sedan', null, null),
       ('Volkswagen', 'GE', 'Hatchback', null, null),
       ('Ford', 'US', 'Minibus', null, 'Detroit'),
       ('Haval', 'CN', 'Minibus', null, null),
       ('Lada', 'RU', 'Sedan', null, null),
       ('Chevrolet', 'US', 'Sedan', null, null),
       ('Toyota', 'JP', 'Wagon', null, null),
       ('CAR', 'CA', 'Wagon', null, null),
       ('AVTO', 'CN', 'Wagon', null, null),
       ('AVTO', 'CA', 'Minibus', null, null)
on conflict do nothing;

insert into l8.characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive)
values ('RIGHT', 'Automatic', 'TSI', 'Gasoline', 'FWD'),
       ('LEFT', 'Manual', 'KE-Jetronic', 'Gasoline', 'RWD'),
       ('LEFT', 'CVT', 'MPI', 'Gasoline', 'FWD'),
       ('LEFT', 'Automatic', 'TSI', 'Diesel', 'FWD'),
       ('LEFT', 'Robotic', 'D4', 'Gasoline', 'FWD'),
       ('LEFT', 'Manual', 'MPI', 'Gasoline', 'AWD'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'RWD'),
       ('LEFT', 'CVT', 'TSI', 'Gasoline', 'FWD'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'FWD'),
       ('LEFT', 'Robotic', 'TSI', 'Diesel', 'AWD'),
       ('RIGHT', 'Automatic', 'MPI', 'Gasoline', 'FWD')
on conflict do nothing;

insert into l8.car(name, manufacturer_id, date_of_issue, price, car_brand_id, characteristics_id)
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

insert into l8.showroom(name, address)
values ('АЦ Фрунзе', 'Фрунзе, 252'),
       ('Восток моторс', 'Большевистская, 276/2'),
       ('АЦ Сибирский тракт', 'Жуковского, 96/2'),
       ('Эксперт НСК', 'Большевистская, 276/1')
on conflict do nothing;

insert into l8.seller(first_name, second_name, showroom_id)
values ('Viktor', 'Vatov', 1),
       ('Dmitry', 'Pelevin', 2),
       ('Ilya', 'Amogusov', 3),
       ('Ivan', 'Kalita', 4)
on conflict do nothing;

insert into l8.available_cars(car_id, showroom_id, quantity)
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

insert into l8.selling(car_id, seller_id, showroom_id, date)
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

-- COMMON PART

create or replace function echo(s int) returns char(30) as
$$
begin
    return s::char(30);
end;
$$
    language 'plpgsql';

create or replace function echo(s char(30)) returns char(30) as
$$
begin
    return s;
end;
$$
    language 'plpgsql';
