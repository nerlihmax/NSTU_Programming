--
-- PostgreSQL database dump
--

-- Dumped from database version 15.2 (Debian 15.2-1.pgdg110+1)
-- Dumped by pg_dump version 15.2 (Debian 15.2-1.pgdg110+1)

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

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: admin
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO admin;

--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: admin
--

COMMENT ON SCHEMA public IS '';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: issued_books; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.issued_books (
    id integer NOT NULL,
    name text NOT NULL,
    reader integer,
    date_of_issue date,
    date_of_return timestamp without time zone,
    CONSTRAINT issued_books_date_of_issue_check CHECK ((date_of_issue < CURRENT_DATE))
);


ALTER TABLE public.issued_books OWNER TO admin;

--
-- Name: issued_books_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.issued_books_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.issued_books_id_seq OWNER TO admin;

--
-- Name: issued_books_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.issued_books_id_seq OWNED BY public.issued_books.id;


--
-- Name: reader; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.reader (
    id integer NOT NULL,
    name text NOT NULL
);


ALTER TABLE public.reader OWNER TO admin;

--
-- Name: reader_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.reader_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.reader_id_seq OWNER TO admin;

--
-- Name: reader_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.reader_id_seq OWNED BY public.reader.id;


--
-- Name: issued_books id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.issued_books ALTER COLUMN id SET DEFAULT nextval('public.issued_books_id_seq'::regclass);


--
-- Name: reader id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.reader ALTER COLUMN id SET DEFAULT nextval('public.reader_id_seq'::regclass);


--
-- Data for Name: issued_books; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.issued_books (id, name, reader, date_of_issue, date_of_return) FROM stdin;
1	Гарри Поттер и кубок огня	1	2021-05-18	2022-01-01 12:00:00
2	Марсианин	2	2022-01-01	\N
3	Сумерки	3	2021-05-18	\N
4	Война и мир	4	2021-05-18	2019-01-01 12:00:00
5	Гарри Поттер и узник Азкабана	1	2021-01-09	2022-01-01 12:00:00
\.


--
-- Data for Name: reader; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.reader (id, name) FROM stdin;
1	Дмитрий Иванов
2	Иван Петров
3	Елена Малышева
4	Алексей Сергеевич
\.


--
-- Name: issued_books_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.issued_books_id_seq', 5, true);


--
-- Name: reader_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.reader_id_seq', 4, true);


--
-- Name: issued_books issued_books_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.issued_books
    ADD CONSTRAINT issued_books_pkey PRIMARY KEY (id);


--
-- Name: reader reader_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.reader
    ADD CONSTRAINT reader_pkey PRIMARY KEY (id);


--
-- Name: issued_books issued_books_reader_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.issued_books
    ADD CONSTRAINT issued_books_reader_fkey FOREIGN KEY (reader) REFERENCES public.reader(id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: admin
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;


--
-- PostgreSQL database dump complete
--

