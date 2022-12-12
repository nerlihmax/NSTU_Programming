drop schema if exists public cascade;
create schema if not exists public;

create table if not exists cities
(
    id   serial primary key,
    name text not null
);

create table if not exists providers
(
    id      serial primary key,
    name    text unique not null,
    city_id integer references cities (id)
);

create table if not exists manufacturers
(
    id      serial primary key,
    name    text unique not null,
    city_id integer references cities (id)
);

create table if not exists porcelain_product_types
(
    id   serial primary key,
    name text unique not null
);

create table if not exists porcelain_products
(
    id                        serial primary key,
    porcelain_product_type_id integer references porcelain_product_types (id),
    manufacturer_id           integer references manufacturers (id),
    provider_id               integer references providers (id),
    price                     integer not null,
    date_of_manufacture       date    not null,
    date_of_supply            date    not null,
    date_of_sale              date,
    is_defective              boolean not null default false
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
       ('Калуга'),
       ('Курган'),
       ('Саранск'),
       ('Смоленск'),
       ('Тамбов'),
       ('Тверь'),
       ('Томск'),
       ('Улан-Удэ'),
       ('Химки'),
       ('Чита'),
       ('Элиста'),
       ('Южно-Сахалинск'),
       ('Якутск'),
       ('Ярославль');

insert into porcelain_product_types (name)
values ('Кружка'),
       ('Тарелка'),
       ('Чашка'),
       ('Салатник'),
       ('Суповник'),
       ('Чайник'),
       ('Блюдце'),
       ('Тарелка для пирога'),
       ('Тарелка для пиццы'),
       ('Тарелка для салата'),
       ('Тарелка для супа'),
       ('Тарелка для фруктов'),
       ('Тарелка для овощей'),
       ('Тарелка для десертов');

insert into providers (name, city_id)
values ('Арт-Пласт', 1),
       ('Вертикаль', 2),
       ('Гранд', 3),
       ('Декор', 4),
       ('Дизайнер', 5),
       ('Домашний декор', 6),
       ('Классика', 7),
       ('Коллекция', 8),
       ('Красная линия', 9),
       ('Любимый дом', 10),
       ('Мастерская', 11),
       ('Миланский', 15),
       ('Миллениум', 16),
       ('Модерн', 17),
       ('Модерн-Классика', 18),
       ('Новое поколение', 19),
       ('Новый стиль', 20),
       ('Олимп', 21),
       ('Орхидея', 22),
       ('Панорама', 23),
       ('Парижский', 24),
       ('Премиум', 25),
       ('Престиж', 26),
       ('Россия', 27),
       ('Серебряный век', 28),
       ('Современник', 29),
       ('Стиль', 30),
       ('Триумф', 31),
       ('Универсал', 32),
       ('Фарфор', 33),
       ('Франция', 34),
       ('Хрусталь', 35),
       ('Шарлотка', 36),
       ('Экспресс', 37);

insert into manufacturers (name, city_id)
values ('Кубаньфарфор', 1),
       ('Мастерская керамики Велес', 2),
       ('Фарфор Сысерти', 3),
       ('Леанза Керамика', 4),
       ('Турина гора', 5),
       ('Речицкий фарфоровый завод', 6),
       ('Керамическая Мануфактура', 7),
       ('Андреапольский фарфоровый завод', 8),
       ('Кисловодский фарфоровый завод Феникс', 9),
       ('Кузнецовский фарфор', 10);

-- create function to fill porcelain_products table like in CourseWorkV144
create or replace function fill_porcelain_products(n integer)
    returns void as
$DO$
declare
    t                       int;
    _porcelain_product_type int;
    _manufacturer           int;
    _provider               int;
    _price                  int;
    _date_of_manufacture    date;
    _date_of_supply         date;
    _date_of_sale           date;
    _is_defective           boolean;
begin
    for i in 1..n
        loop
            _porcelain_product_type = (select id from porcelain_product_types order by random() limit 1);
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
            if (random() > 0.95) then
                _is_defective = true;
            else
                _is_defective = false;
            end if;
            insert into porcelain_products (porcelain_product_type_id, manufacturer_id, provider_id, price,
                                            date_of_manufacture, date_of_supply, date_of_sale, is_defective)
            values (_porcelain_product_type, _manufacturer, _provider, _price, _date_of_manufacture, _date_of_supply,
                    _date_of_sale, _is_defective);
        end loop;
end;
$DO$ language plpgsql;

select fill_porcelain_products(1000);