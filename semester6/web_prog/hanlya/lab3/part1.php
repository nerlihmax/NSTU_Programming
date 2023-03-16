<?php

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();

$books = pg_query(
    $db,
    "
    select book.id,
    book.name,
    reader.name as reader_name,
    book.date_of_issue,
    book.date_of_return
    from issued_books as book
    inner join reader on reader.id = book.reader "
);

?>

<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <script src="https://cdn.tailwindcss.com"></script>
    <title>Лабораторная №3</title>
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
    <div class="flex flex-col h-full w-full justify-top items-center py-10 space-y-10 my-10">
        <div class="flex flex-col bg-white w-4/6 rounded-xl justify-center items-center space-y-4">
            <table>
                <tr>
                    <th>ID</th>
                    <th>Название</th>
                    <th>Читатель</th>
                    <th>Дата выдачи</th>
                    <th>Дата возврата</th>
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
                }
                ?>

                </tr>
            </table>
        </div>

        <style>
            #v-legend {
                display: inline-block;
                transform: rotateZ(-90deg);
                position: absolute;
                top: 50%;
                left: -20%;
            }

            .graph {
                position: relative;
                max-width: fit-content;
            }
        </style>

        <div class="graph">
            <h3 class="text-xl" id="v-legend">Количество книг у читателя</h3>
            <img id="image" src="chart.php" alt="">
        </div>

        <h3 class="text-xl">Читатель</h3>
        <br>
        <br>
        <script>
            setInterval(() => {
                image.src = 'chart.php?' + new Date().getTime();
            }, 1000);
        </script>
    </div>
</body>

</html>