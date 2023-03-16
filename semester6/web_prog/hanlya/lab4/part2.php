<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();

$query = 'select id, name from reader';

$result = pg_query($db, $query);

$readers = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($readers, $line);
}
?>
<html>

<head>
    <meta charset="UTF-8">
    <title>Часть 2</title>
    <style>
        :root {
            --main-color: #575757;
        }

        table {
            border-collapse: collapse;
            border-color: black;
        }

        table,
        th,
        td {
            border: 1px solid var(--main-color);
        }

        th,
        td {
            padding: 1rem;
        }

        th {
            background: var(--main-color);
            color: white;
        }

        td {
            color: #666;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №4</h1>
    <h1>Вариант 15</h1>
    <h2>Часть 2</h2>

    <h3>Выберите читателя</h3>
    <form class="form">
        <select class="readers" name="reader">
            <?php
            foreach ($readers as $value) {
                echo "<option value=\"" . $value['id'] . "\">" . $value['name'] . "</option>";
            }
            ?>
        </select>
    </form>

    <br>

    <h3>Книги читателя</h3>

    <table class="list" border="1">
        <tr>
            <th>ID</th>
            <th>Название</th>
            <th>Читатель</th>
            <th>Дата выдачи</th>
            <th>Дата возврата</th>
        <tr>
    </table>

    <script>
        const getBooksByReaderName = async (readerId) => {
            const response = await fetch(`https://hanlya.x.kheynov.ru/lab4/books_by_reader.php?reader=` + readerId);
            const json = await response.json();
            console.log(json);
            return json;
        };

        const renderList = ({ books }) => {
            const list = document.querySelector('.list');

            [...list.children].forEach((child, idx) => idx > 0 && child.remove());

            books
                .map((book) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>${book.id}</td> 
                    <td>${book.name}</td> 
                    <td>${book.reader_name}</td> 
                    <td>${book.date_of_issue}</td> 
                    <td>${book.date_of_return}</td`;
                    return row;
                })
                .forEach((row) => list.appendChild(row));
        };

        const select = document.querySelector('.readers');

        const app = async () => {
            const book = select.value;
            const books = await getBooksByReaderName(book);
            renderList({ books });
        };

        app();

        select.addEventListener('input', app);
    </script>

</body>

</html>