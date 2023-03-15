<?php

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();

session_start();

$books = pg_query(
    $db, "
    select book.id,
    book.name,
    reader.name as reader_name,
    book.date_of_issue,
    book.date_of_return
    from issued_books as book
    inner join reader on reader.id = book.reader 
    where " . (isset($_GET['reader_name']) ? "lower(reader.name) like lower('%" . pg_escape_string($_GET['reader_name']) . "%')" : "1=1")
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
    <div class="flex flex-col h-full w-full justify-center items-center py-10 space-y-10 my-10">
        <div class="flex flex-col bg-white w-4/6 rounded-xl justify-center items-center space-y-4">
            <div class="flex flex-row w-1/2 h-full justify-between py-10 items-center">
                <?php
                if ($authenticated) {
                    echo '<form method="post" action="index.php"><p><button type="submit" class="px-6 py-3 rounded bg-amber-400" name="log_out">Выход</button></p></form>';
                } else {
                    echo '<form method="get" action="auth.php"><p><button type="submit" class="px-6 py-3 rounded bg-amber-400">Вход</button></p></form>';
                }
                ?>
                <span>
                    <?= isset($_SESSION['login']) ? 'Текущий пользователь: ' . $_SESSION['login'] : '' ?>
                </span>
            </div>


            <h1 class="text-xl">Поиск по читателю: </h1>
            <form action="index.php" method="get" class="flex flex-row w-full justify-center items-center space-x-4">
                <input name="reader_name" type="text" class="px-4 py-2 rounded bg-slate-600 text-white">
                <button type="submit" class="px-4 py-2 bg-amber-400 rounded">Поиск</button>
            </form>
            <br>
            <?php
            if ($authenticated) {
                echo '<a href="insert.php" class="text-black px-6 py-3 bg-amber-400 rounded">Добавить запись</a>';
            }
            ?>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Название</th>
                    <th>Читатель</th>
                    <th>Дата выдачи</th>
                    <th>Дата возврата</th>
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