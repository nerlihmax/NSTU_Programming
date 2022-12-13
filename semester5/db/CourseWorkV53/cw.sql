drop schema if exists public cascade;
create schema if not exists public;

create table if not exists countries
(
    id   serial primary key,
    name text not null
);

create table if not exists providers
(
    id         serial primary key,
    name       text unique not null,
    country_id integer references countries (id)
);

create table if not exists manufacturers
(
    id         serial primary key,
    name       text unique not null,
    country_id integer references countries (id)
);

create table plumbing_types
(
    id   serial primary key,
    name text unique not null
);

create table clients
(
    id   serial primary key,
    name text unique not null
);

create table if not exists plumbings
(
    id                  serial primary key,
    type                integer references plumbing_types (id),
    manufacturer_id     integer references manufacturers (id),
    provider_id         integer references providers (id),
    price               integer not null,
    date_of_manufacture date    not null,
    date_of_supply      date    not null
);

create table if not exists sales
(
    id          serial primary key,
    client_id   integer references clients (id),
    plumbing_id integer references plumbings (id),
    date        date    not null,
    quantity    integer not null
);

insert into countries(name)
values ('Russia'),
       ('USA'),
       ('China'),
       ('Germany'),
       ('France'),
       ('Italy'),
       ('Spain'),
       ('UK'),
       ('Japan'),
       ('Canada'),
       ('Brazil'),
       ('Australia'),
       ('India'),
       ('Mexico'),
       ('Indonesia'),
       ('Netherlands'),
       ('Turkey'),
       ('Switzerland'),
       ('Belgium'),
       ('Sweden'),
       ('Poland'),
       ('Austria'),
       ('Norway'),
       ('Denmark'),
       ('Ireland'),
       ('Czech Republic'),
       ('Portugal');

insert into providers(name, country_id)
values ('СанТехСити', 1),
       ('100 градусов', 2),
       ('Галлоп', 3),
       ('Енот', 4),
       ('Флорентина Сибирь', 5),
       ('Сантехопт', 6),
       ('Сантаком', 7),
       ('Меркурий Импорт', 8),
       ('Сантел-м', 9),
       ('Атмосфера тепла', 10),
       ('Чешский двор', 11),
       ('Формула М2', 12);

insert into manufacturers(name, country_id)
values ('Faucet', 1),
       ('Kohler', 2),
       ('Moen', 3),
       ('Delta', 4),
       ('Toto', 5),
       ('Union', 5),
       ('American Standard', 6);

insert into plumbing_types(name)
values ('Трубы'),
       ('Фильтры'),
       ('Сифоны'),
       ('Краны'),
       ('Полотенцесушители'),
       ('Клапаны'),
       ('Бачки'),
       ('Биде'),
       ('Душевые кабины'),
       ('Душевые гарнитуры'),
       ('Душевые стойки'),
       ('Душевые штанги'),
       ('Комплектующие для душевых штанг'),
       ('Комплектующие для душевых гарнитур'),
       ('Комплектующие для душевых кабин'),
       ('Комплектующие для душевых стоек'),
       ('Душевые аксессуары'),
       ('Душевые боксы'),
       ('Душевые капища'),
       ('Душевые накладки'),
       ('Душевые панели'),
       ('Душевые подставки'),
       ('Душевые фильтры'),
       ('Душевые шланги'),
       ('Комплектующие для душевых боксов'),
       ('Комплектующие для душевых капищ'),
       ('Комплектующие для душевых накладок'),
       ('Комплектующие для душевых панелей'),
       ('Комплектующие для душевых подставок'),
       ('Комплектующие для душевых фильтров'),
       ('Комплектующие для душевых шлангов');

insert into clients(name)
values ('ИП Василий Петров'),
       ('ИП Александр Сидоров'),
       ('ИП Алексей Иванов'),
       ('ИП Андрей Кузнецов'),
       ('ИП Антон Смирнов'),
       ('ИП Артем Соколов'),
       ('ИП Артур Лебедев'),
       ('ИП Борис Козлов'),
       ('ИП Вадим Новиков'),
       ('ИП Валерий Морозов'),
       ('ИП Василий Попов'),
       ('ИП Виктор Лебедев'),
       ('ИП Виталий Семенов'),
       ('ИП Владимир Егоров'),
       ('ИП Владислав Козлов'),
       ('ИП Владислав Кузнецов'),
       ('ИП Владислав Смирнов'),
       ('ИП Владислав Соколов'),
       ('ИП Владислав Соловьев'),
       ('ИП Владислав Сорокин'),
       ('ИП Владислав Тарасов'),
       ('ИП Владислав Чернов'),
       ('ИП Владислав Шестаков'),
       ('ИП Владислав Ширяев'),
       ('ИП Владислав Щербаков'),
       ('ИП Владислав Яковлев'),
       ('ИП Владлен Кузнецов'),
       ('ИП Владлен Смирнов'),
       ('ИП Владлен Соколов'),
       ('ИП Владлен Соловьев'),
       ('ИП Владлен Сорокин'),
       ('ИП Владлен Тарасов'),
       ('ИП Владлен Чернов'),
       ('ИП Владлен Шестаков');

create or replace function fill_plumbing_products(n integer)
    returns void as
$DO$
declare
    t                    int;
    _type                int;
    _manufacturer        int;
    _provider            int;
    _price               int;
    _date_of_manufacture date;
    _date_of_supply      date;
    _date_of_sale        date;
begin
    for i in 1..n
        loop
            _type = (select id from plumbing_types order by random() limit 1);
            _manufacturer = (select id from manufacturers order by random() limit 1);
            _provider = (select id from providers order by random() limit 1);
            _price = (select random() * 10000)::int;
            _date_of_manufacture = '2018-01-01'::date + (select random() * 365)::int;
            _date_of_supply = _date_of_manufacture + (select random() * 365)::int;
            if (random() > 0.5) then
                _date_of_sale = _date_of_supply + (select random() * 365)::int;
            else
                _date_of_sale = null;
            end if;

            insert into plumbings (type, manufacturer_id, provider_id, price, date_of_manufacture, date_of_supply)
            values (_type, _manufacturer, _provider, _price, _date_of_manufacture, _date_of_supply);
        end loop;
end;
$DO$ language plpgsql;

create or replace function fill_sales(n integer)
    returns void as
$DO$
declare
    t             int;
    _plumbing     int;
    _client_id    int;
    _date_of_sale date;
    _quantity     int;
begin
    for i in 1..n
        loop
            _plumbing = (select id from plumbings order by random() limit 1);
            _date_of_sale = '2018-01-01'::date + (select random() * 365)::int;
            _client_id = (select id from clients order by random() limit 1);
            _quantity = (select random() * 100)::int;
            insert into sales (client_id, plumbing_id, date, quantity)
            values (_client_id, _plumbing, _date_of_sale, _quantity);
        end loop;
end;
$DO$ language plpgsql;

select fill_plumbing_products(1000);
select fill_sales(250);

drop role if exists operator;
drop role if exists db_user;
drop role if exists admin;

--Пересоздаем пользователей
create role operator with login password '12345';
create role db_user with login password '12345';
create role admin with login password '12345';

--Выделяем права на схему public
grant usage on schema public to operator;
grant usage on schema public to db_user;
grant usage on schema public to admin;

grant all on schema public to admin;
grant all on schema public to operator;
grant all on schema public to db_user;

grant select, usage on all sequences in schema public to operator;
grant select, usage on all sequences in schema public to db_user;
grant all privileges on all sequences in schema public to admin;

--Выдаем права на таблицы справочники оператору, на остальные таблицы обычному пользователю, администратору на все
grant insert, select, update, delete on countries, manufacturers, plumbing_types to operator;
grant insert, select, update, delete on plumbings, providers, clients, sales to db_user;
grant all privileges on all tables in schema public to admin;