<?php

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();

session_start();

$books = pg_query(
    $db,
    "SELECT 
        book.id, 
        book.title,
        book.year,
        book.price,
        a.name    AS author,
        p.name    AS publisher,
        s.name    AS seller,
        a.address AS author_address,
        p.address AS publisher_address
    FROM book
        INNER JOIN author a ON a.id = book.author_id
        INNER JOIN publisher p ON book.publisher_id = p.id
        INNER JOIN seller s ON s.id = book.seller_id
    WHERE " . (isset($_GET['author']) ? "lower(a.name) like lower('%" . pg_escape_string($_GET['author']) . "%')" : "1=1")
);

if (isset($_POST["log_out"])) {
    session_destroy();
    header("location:index.php");
}

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (isset($_POST["log_out"])) {
    session_destroy();
    header("location:index.php");
}

?>

<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <script src="https://cdn.tailwindcss.com"></script>
    <title>Лабораторная №2</title>
    <style type="text/tailwindcss">
        @tailwind base;
        @tailwind components;
    </style>
    <style>
        :root {
            --main-color: #10b981;
        }

        form {
            padding: 0;
            margin: 0;
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
    <div class="flex flex-col h-full w-full justify-top items-center space-y-10">
        <div class="flex flex-row w-full bg-gray-200 h-full justify-center items-center space-x-10">
            <div class="flex flex-col w-full justify-center items-end my-3">
                <?php
                if ($authenticated) {
                    ?>

                    <form method="post" action="index.php" class="px-6 py-3 rounded-full bg-amber-400 text-center">
                        <button type="submit" name="log_out">Выход</button>
                    </form>

                    <?php
                } else {
                    ?>
                    <form method="get" action="auth.php" class="px-6 py-3 rounded-full bg-amber-400 text-center">
                        <button type="submit">Вход</button>
                    </form>
                    <?php
                }
                ?>
            </div>
            <p>
                <?= isset($_SESSION['login']) ? 'Уровень доступа: ' . ($_SESSION['group'] < 2 ? 'Чтение, добавление, редактирование' : 'Чтение, добавление, удаление, редактирование') : '' ?>
            </p>
        </div>
        <div class="flex flex-col bg-white w-4/6 rounded-xl justify-center items-center space-y-4">

            <form action="index.php" method="get" class="flex flex-row w-full justify-center items-center space-x-4">
                <div class="w-72">
                    <div class="relative h-10  w-full min-w-[200px]">
                        <input name="author"
                            class="peer h-full w-full rounded-[7px] border border-blue-gray-200 border-t-transparent bg-transparent px-3 py-2.5 font-sans text-sm font-normal text-blue-gray-700 outline outline-0 transition-all placeholder-shown:border placeholder-shown:border-blue-gray-200 placeholder-shown:border-t-blue-gray-200 focus:border-2 focus:border-amber-400 focus:border-t-transparent focus:outline-0 disabled:border-0 disabled:bg-blue-gray-50"
                            placeholder=" " />
                        <label
                            class="before:content[' '] after:content[' '] pointer-events-none absolute left-0 -top-1.5 flex h-full w-full select-none text-[11px] font-normal leading-tight text-blue-gray-400 transition-all before:pointer-events-none before:mt-[6.5px] before:mr-1 before:box-border before:block before:h-1.5 before:w-2.5 before:rounded-tl-md before:border-t before:border-l before:border-blue-gray-200 before:transition-all after:pointer-events-none after:mt-[6.5px] after:ml-1 after:box-border after:block after:h-1.5 after:w-2.5 after:flex-grow after:rounded-tr-md after:border-t after:border-r after:border-blue-gray-200 after:transition-all peer-placeholder-shown:text-sm peer-placeholder-shown:leading-[3.75] peer-placeholder-shown:text-blue-gray-500 peer-placeholder-shown:before:border-transparent peer-placeholder-shown:after:border-transparent peer-focus:text-[11px] peer-focus:leading-tight peer-focus:text-amber-400 peer-focus:before:border-t-2 peer-focus:before:border-l-2 peer-focus:before:border-amber-400 peer-focus:after:border-t-2 peer-focus:after:border-r-2 peer-focus:after:border-amber-400 peer-disabled:text-transparent peer-disabled:before:border-transparent peer-disabled:after:border-transparent peer-disabled:peer-placeholder-shown:text-blue-gray-500">
                            Поиск по автору
                        </label>
                    </div>
                </div>

                <button type="submit" class="px-4 py-2 bg-amber-400 rounded-full">Поиск</button>
            </form>
            <br>
            <?php
            if ($authenticated) {
                echo '<a href="insert.php" class="text-black px-6 py-3 bg-amber-400 rounded-full">Добавить запись</a>';
            }
            ?>
            <table cla>
                <tr>
                    <th>ID</th>
                    <th>Название</th>
                    <th>Год издания</th>
                    <th>Цена, $</th>
                    <th>Автор</th>
                    <th>Издатель</th>
                    <th>Продавец</th>
                    <th>Адрес автора</th>
                    <th>Адрес издателя</th>

                    <?php
                    if ($authenticated && $_SESSION['group'] >= 1) {
                        echo "<th></th>";
                    }
                    if ($authenticated && $_SESSION['group'] >= 2) {
                        echo "<th></th>";
                    }
                    ?>
                </tr>
                <?php
                while ($line = pg_fetch_array($books, null, PGSQL_ASSOC)) {
                    echo "\t<tr>\n";
                    foreach ($line as $col_value) {
                        ?>
                        <td>
                            <?= $col_value ?>
                        </td>
                        <?php
                    }
                    if ($authenticated && $_SESSION['group'] >= 1) {
                        echo "\t<td><a class=\"underline text-blue-500\" href=\"update.php?id=" . $line["id"] . "\">Update</a></td>";
                        if ($authenticated && $_SESSION['group'] >= 2) {
                            echo "\t<td><a class=\"underline text-blue-500\" href=\"delete.php?id=" . $line["id"] . "\">Delete</a></td>";
                        }
                    }
                }
                ?>
                </tr>
            </table>
        </div>
    </div>
</body>

</html>