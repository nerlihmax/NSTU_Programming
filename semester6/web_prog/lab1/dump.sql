--
-- PostgreSQL database dump
--

-- Dumped from database version 14.4 (Debian 14.4-1.pgdg110+1)
-- Dumped by pg_dump version 14.4 (Debian 14.4-1.pgdg110+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: ad_types; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.ad_types (
    id integer NOT NULL,
    type text
);


ALTER TABLE public.ad_types OWNER TO admin;

--
-- Name: ad_types_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.ad_types_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ad_types_id_seq OWNER TO admin;

--
-- Name: ad_types_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.ad_types_id_seq OWNED BY public.ad_types.id;


--
-- Name: ads; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.ads (
    id integer NOT NULL,
    type integer,
    city integer,
    address text,
    roominess integer,
    price integer,
    created_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.ads OWNER TO admin;

--
-- Name: ads_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.ads_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ads_id_seq OWNER TO admin;

--
-- Name: ads_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.ads_id_seq OWNED BY public.ads.id;


--
-- Name: cities; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.cities (
    id integer NOT NULL,
    name text
);


ALTER TABLE public.cities OWNER TO admin;

--
-- Name: cities_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.cities_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cities_id_seq OWNER TO admin;

--
-- Name: cities_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.cities_id_seq OWNED BY public.cities.id;


--
-- Name: ad_types id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.ad_types ALTER COLUMN id SET DEFAULT nextval('public.ad_types_id_seq'::regclass);


--
-- Name: ads id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.ads ALTER COLUMN id SET DEFAULT nextval('public.ads_id_seq'::regclass);


--
-- Name: cities id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.cities ALTER COLUMN id SET DEFAULT nextval('public.cities_id_seq'::regclass);


--
-- Data for Name: ad_types; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.ad_types (id, type) FROM stdin;
1	Сниму
2	Сдам
3	Продам
4	Куплю
\.


--
-- Data for Name: ads; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.ads (id, type, city, address, roominess, price, created_at) FROM stdin;
1	1	1	ул. Красная, 1	2	20000	2023-02-12 14:28:41.132754
2	2	2	ул. Синяя, 2	1	20000	2023-02-12 14:28:41.132754
3	3	3	ул. Зеленая, 3	3	8000000	2023-02-12 14:28:41.132754
4	4	4	ул. Карла Маркса, 2	4	40000000	2023-02-12 14:28:41.132754
5	1	5	ул. Зорге, 21	2	25000	2023-02-12 14:28:41.132754
6	4	4	ул. Желтая, 4	4	40000	2023-02-12 14:28:41.132754
\.


--
-- Data for Name: cities; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.cities (id, name) FROM stdin;
1	Санкт-Петербург
2	Москва
3	Краснодар
4	Владивосток
5	Новосибирск
\.


--
-- Name: ad_types_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.ad_types_id_seq', 4, true);


--
-- Name: ads_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.ads_id_seq', 6, true);


--
-- Name: cities_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.cities_id_seq', 5, true);


--
-- Name: ad_types ad_types_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.ad_types
    ADD CONSTRAINT ad_types_pkey PRIMARY KEY (id);


--
-- Name: ad_types ad_types_type_key; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.ad_types
    ADD CONSTRAINT ad_types_type_key UNIQUE (type);


--
-- Name: ads ads_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.ads
    ADD CONSTRAINT ads_pkey PRIMARY KEY (id);


--
-- Name: cities cities_name_key; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.cities
    ADD CONSTRAINT cities_name_key UNIQUE (name);


--
-- Name: cities cities_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.cities
    ADD CONSTRAINT cities_pkey PRIMARY KEY (id);


--
-- Name: ads ads_city_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.ads
    ADD CONSTRAINT ads_city_fkey FOREIGN KEY (city) REFERENCES public.cities(id);


--
-- Name: ads ads_type_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.ads
    ADD CONSTRAINT ads_type_fkey FOREIGN KEY (type) REFERENCES public.ad_types(id);


--
-- PostgreSQL database dump complete
--

