drop schema if exists public cascade;
create schema if not exists public;

create table cities -- города
(
    id   serial primary key,
    name text unique
);

create table manufacturers -- производители
(
    id      serial primary key,
    name    text unique,
    city_id int references cities (id)
);

create table providers -- поставщики
(
    id   serial primary key,
    name text unique
);

create table servings -- виды мороженого
(
    id      serial primary key,
    serving text unique
);

create table tastes
(
    id    serial primary key,
    taste text unique
);

create table ice_creams -- мороженое
(
    id              serial primary key,
    serving         int references servings (id),
    taste           int references tastes (id),
    manufacturer_id int references manufacturers (id),
    provider_id     int references providers (id),
    price           int check (price > 0),
    weight          int check (weight > 50),
    date_of_issue   date,
    date_of_supply  date,
    date_of_sale    date,
    is_defective    boolean
);

insert into cities (name)
values ('Москва'),
       ('Санкт-Петербург'),
       ('Казань'),
       ('Новосибирск'),
       ('Екатеринбург'),
       ('Нижний Новгород'),
       ('Красноярск'),
       ('Самара'),
       ('Омск'),
       ('Челябинск'),
       ('Ростов-на-Дону'),
       ('Уфа'),
       ('Краснодар'),
       ('Пермь'),
       ('Волгоград'),
       ('Воронеж'),
       ('Саратов'),
       ('Киров'),
       ('Тюмень'),
       ('Тольятти'),
       ('Ижевск'),
       ('Барнаул'),
       ('Ульяновск'),
       ('Иркутск'),
       ('Хабаровск'),
       ('Ярославль'),
       ('Владивосток'),
       ('Махачкала'),
       ('Томск'),
       ('Оренбург'),
       ('Кемерово'),
       ('Новокузнецк'),
       ('Рязань'),
       ('Астрахань'),
       ('Пенза'),
       ('Липецк'),
       ('Калининград'),
       ('Тула'),
       ('Курск'),
       ('Брянск'),
       ('Иваново'),
       ('Владимир'),
       ('Сочи'),
       ('Чебоксары'),
       ('Калуга'),
       ('Ставрополь'),
       ('Белгород'),
       ('Сургут'),
       ('Симферополь'),
       ('Севастополь'),
       ('Архангельск'),
       ('Вологда'),
       ('Кострома'),
       ('Курган'),
       ('Саранск'),
       ('Смоленск'),
       ('Тамбов'),
       ('Тверь'),
       ('Улан-Удэ'),
       ('Химки'),
       ('Чита'),
       ('Элиста'),
       ('Южно-Сахалинск'),
       ('Якутск');

insert into manufacturers(name, city_id)
values ('Останкинский молочный комбинат', (select id from cities order by random() limit 1)),
       ('Морозофф', (select id from cities order by random() limit 1)),
       ('Волгомясомолторг', (select id from cities order by random() limit 1)),
       ('Башкирский Холод', (select id from cities order by random() limit 1)),
       ('Комос Групп', (select id from cities order by random() limit 1)),
       ('Чистая линия', (select id from cities order by random() limit 1)),
       ('Вилон', (select id from cities order by random() limit 1)),
       ('Молокозавод Поронайский', (select id from cities order by random() limit 1)),
       ('Дакгомз', (select id from cities order by random() limit 1)),
       ('Актиформула', (select id from cities order by random() limit 1)),
       ('Ангария', (select id from cities order by random() limit 1));

insert into providers(name)
values ('Полярис'),
       ('Новосибирский хладокомбинат'),
       ('Гулливер'),
       ('Восток-Запад'),
       ('Вишера-Плюс'),
       ('Юнилевер Русь'),
       ('Север'),
       ('Десант здоровья'),
       ('Любимая Зима'),
       ('Ангария'),
       ('Гроспирон');

insert into servings(serving)
-- insert ice cream types in russian
values ('Рожок'),
       ('Стаканчик'),
       ('Эскимо'),
       ('Рулет'),
       ('Торт'),
       ('В шариках'),
       ('На палочке');

insert into tastes (taste)
values ('Клубничный'),
       ('Шоколадный'),
       ('Пломбир'),
       ('Крем-брюле'),
       ('Апельсиновый'),
       ('Черничный'),
       ('В глазури');

-- create function for filling ice_cream table
create or replace function fill_ice_creams_table(n int)
    returns void as
$$
declare
    i                integer := 0;
    _serving         integer;
    _taste           integer;
    _manufacturer_id integer;
    _provider_id     integer;
    _price           int;
    _weight          int;
    _date_of_issue   date;
    _date_of_supply  date;
    _date_of_sale    date;
    _is_defective    boolean;
begin
    while i < n
        loop
            _serving := (select id from servings order by random() limit 1);
            _taste := (select id from tastes order by random() limit 1);
            _manufacturer_id := (select id from manufacturers order by random() limit 1);
            _provider_id := (select id from providers order by random() limit 1);
            _price = (round(random() * 100 + 2))::int;
            _weight = (round(random() * 100 + 80))::int;
            _date_of_issue = '2020-01-01'::date + (select random() * 365)::int;
            _date_of_supply = _date_of_issue + (select random() * 365)::int;
            if (random() > 0.5) then
                _date_of_sale = _date_of_supply + (select random() * 365)::int;
            else
                _date_of_sale = null;
            end if;
            if (random() > 0.95) then
                _is_defective = true;
            else
                _is_defective = false;
            end if;
            insert into ice_creams(serving, taste, manufacturer_id, provider_id, price, weight, date_of_issue,
                                   date_of_supply, date_of_sale, is_defective)
            values (_serving, _taste, _manufacturer_id, _provider_id, _price, _weight, _date_of_issue,
                    _date_of_supply, _date_of_sale, _is_defective);
            i := i + 1;
        end loop;
end;
$$ language plpgsql;

select fill_ice_creams_table(1000);

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
grant insert, select, update, delete on cities, manufacturers, servings, tastes to operator;
grant insert, select, update, delete on ice_creams, providers to db_user;
grant all privileges on all tables in schema public to admin;