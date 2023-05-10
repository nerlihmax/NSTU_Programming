<html>

<head>
    <title>Лабораторная №3</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №3</h1>
    <h2>Вариант 12</h2>
    <h2>Часть 1</h2>


    <?php
    session_start();
    require_once(__DIR__ . '/../../utils/connection.php');
    $db = connect('web_v12');

    $documents = pg_query(
        $db,
        "SELECT 
            d.id,
            w.name as worker,
            d.name as document,
            d.date_of_apply,
            d.date_of_return
        from documents as d
            inner join workers w on w.id = d.worker;"
    );

    ?>
    <table border="1">
        <tr>
            <th>ID</th>
            <th>Исполнитель</th>
            <th>Договор</th>
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
        while ($line = pg_fetch_array($documents, null, PGSQL_ASSOC)) {
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

    <br>
    <div>
        <h3>Количество документов по исполнителям</h3>
        <br>
        <img id="image" src="chart.php" alt="">
    </div>

    <script>
        setInterval(() => {
            image.src = 'chart.php?' + new Date().getTime();
        }, 1000);
    </script>
</body>

</html>