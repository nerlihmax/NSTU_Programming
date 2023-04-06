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
-- Name: author; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.author (
    id integer NOT NULL,
    name text NOT NULL,
    address text NOT NULL
);


ALTER TABLE public.author OWNER TO admin;

--
-- Name: author_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.author_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.author_id_seq OWNER TO admin;

--
-- Name: author_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.author_id_seq OWNED BY public.author.id;


--
-- Name: book; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.book (
    id integer NOT NULL,
    title text NOT NULL,
    author_id integer,
    publisher_id integer,
    year integer NOT NULL,
    price real NOT NULL,
    seller_id integer
);


ALTER TABLE public.book OWNER TO admin;

--
-- Name: book_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.book_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.book_id_seq OWNER TO admin;

--
-- Name: book_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.book_id_seq OWNED BY public.book.id;


--
-- Name: publisher; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.publisher (
    id integer NOT NULL,
    name text NOT NULL,
    address text NOT NULL
);


ALTER TABLE public.publisher OWNER TO admin;

--
-- Name: publisher_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.publisher_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.publisher_id_seq OWNER TO admin;

--
-- Name: publisher_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.publisher_id_seq OWNED BY public.publisher.id;


--
-- Name: seller; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.seller (
    id integer NOT NULL,
    name text NOT NULL
);


ALTER TABLE public.seller OWNER TO admin;

--
-- Name: seller_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.seller_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.seller_id_seq OWNER TO admin;

--
-- Name: seller_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.seller_id_seq OWNED BY public.seller.id;


--
-- Name: author id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.author ALTER COLUMN id SET DEFAULT nextval('public.author_id_seq'::regclass);


--
-- Name: book id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.book ALTER COLUMN id SET DEFAULT nextval('public.book_id_seq'::regclass);


--
-- Name: publisher id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.publisher ALTER COLUMN id SET DEFAULT nextval('public.publisher_id_seq'::regclass);


--
-- Name: seller id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.seller ALTER COLUMN id SET DEFAULT nextval('public.seller_id_seq'::regclass);


--
-- Data for Name: author; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.author (id, name, address) FROM stdin;
1	J. K. Rowling	London, UK
2	J. R. R. Tolkien	Bournmouth, UK
3	Stephen King	Portland, ME, USA
4	George R. R. Martin	Bayonne, NJ, USA
5	Dan Brown	Exeter, NH, USA
6	J. D. Salinger	New York, NY, USA
7	J. R. R. Martin	Bayonne, NJ, USA
8	J. D. Robb	New York, NY, USA
9	J. K. Rowling	London, UK
10	J. R. R. Tolkien	Bournmouth, UK
11	Stephen King	Portland, ME, USA
12	George R. R. Martin	Bayonne, NJ, USA
13	Dan Brown	Exeter, NH, USA
14	J. D. Salinger	New York, NY, USA
15	J. R. R. Martin	Bayonne, NJ, USA
16	J. D. Robb	New York, NY, USA
\.


--
-- Data for Name: book; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.book (id, title, author_id, publisher_id, year, price, seller_id) FROM stdin;
1	Harry Potter and the Philosopher's Stone	1	1	1997	8.99	1
2	Harry Potter and the Chamber of Secrets	1	1	1998	8.99	1
3	Harry Potter and the Prisoner of Azkaban	1	1	1999	8.99	1
4	Harry Potter and the Goblet of Fire	1	1	2000	8.99	1
5	Harry Potter and the Order of the Phoenix	1	1	2003	8.99	1
6	Harry Potter and the Half-Blood Prince	1	1	2005	8.99	1
7	Harry Potter and the Deathly Hallows	1	1	2007	8.99	1
8	The Fellowship of the Ring	2	2	1954	8.99	2
9	The Two Towers	2	2	1954	8.99	2
10	The Return of the King	2	2	1955	8.99	2
11	The Shining	3	3	1977	8.99	3
12	The Stand	3	3	1978	8.99	3
13	It	3	3	1986	8.99	3
14	The Gunslinger	4	4	1982	8.99	4
15	The Drawing of the Three	4	4	1987	8.99	4
16	The Waste Lands	4	4	1991	8.99	4
17	The Dark Tower	4	4	2003	8.99	4
18	The Da Vinci Code	5	5	2003	8.99	5
19	Angels & Demons	5	5	2000	8.99	5
20	The Lost Symbol	5	5	2009	8.99	5
21	The Catcher in the Rye	6	6	1951	8.99	6
22	The Hobbit	7	7	1937	8.99	6
23	The Silmarillion	7	7	1977	8.99	6
24	A Game of Thrones	8	8	1996	8.99	6
25	A Clash of Kings	8	8	1998	8.99	6
26	A Storm of Swords	8	8	2000	8.99	6
27	A Feast for Crows	8	8	2005	8.99	6
28	A Dance with Dragons	8	8	2011	8.99	6
\.


--
-- Data for Name: publisher; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.publisher (id, name, address) FROM stdin;
1	Bloomsbury	London, UK
2	Allen & Unwin	Sydney, Australia
3	Houghton Mifflin	Boston, MA, USA
4	HarperCollins	New York, NY, USA
5	Bloomsbury	London, UK
6	Allen & Unwin	Sydney, Australia
7	Houghton Mifflin	Boston, MA, USA
8	HarperCollins	New York, NY, USA
\.


--
-- Data for Name: seller; Type: TABLE DATA; Schema: public; Owner: admin
--

COPY public.seller (id, name) FROM stdin;
1	Amazon
2	Barnes & Noble
3	Books-A-Million
4	Amazon
5	Barnes & Noble
6	Books-A-Million
\.


--
-- Name: author_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.author_id_seq', 16, true);


--
-- Name: book_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.book_id_seq', 28, true);


--
-- Name: publisher_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.publisher_id_seq', 8, true);


--
-- Name: seller_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.seller_id_seq', 6, true);


--
-- Name: author author_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.author
    ADD CONSTRAINT author_pkey PRIMARY KEY (id);


--
-- Name: book book_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.book
    ADD CONSTRAINT book_pkey PRIMARY KEY (id);


--
-- Name: publisher publisher_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.publisher
    ADD CONSTRAINT publisher_pkey PRIMARY KEY (id);


--
-- Name: seller seller_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.seller
    ADD CONSTRAINT seller_pkey PRIMARY KEY (id);


--
-- Name: book book_author_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.book
    ADD CONSTRAINT book_author_id_fkey FOREIGN KEY (author_id) REFERENCES public.author(id);


--
-- Name: book book_publisher_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.book
    ADD CONSTRAINT book_publisher_id_fkey FOREIGN KEY (publisher_id) REFERENCES public.publisher(id);


--
-- Name: book book_seller_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.book
    ADD CONSTRAINT book_seller_id_fkey FOREIGN KEY (seller_id) REFERENCES public.seller(id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: admin
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;


--
-- PostgreSQL database dump complete
--

