<html>

<head>
    <title>Лабораторная №3</title>
</head>

<body>
    <h1>Лабораторная №3</h1>
    <h2>Вариант 11</h2>

    <?php
    session_start();
    require_once(__DIR__ . '/../../utils/connection.php');
    $db = connect('web_v31');

    $detail = pg_query(
        $db,
        "SELECT 
            map.id,
            map.name,
            o.name as operation,
            duration
        from technological_map as map
          inner join operation o on o.id = map.operation"
    );
    ?>

    <br>
    <br>
    <table border="1">
        <tr>
            <th>ID</th>
            <th>Название детали</th>
            <th>Вид операции</th>
            <th>Длительность операции в минутах</th>
        </tr>
        <?php
        while ($line = pg_fetch_array($detail, null, PGSQL_ASSOC)) {
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
    <h3>Количество обработанных деталей по виду обработки</h3>
    <br>
    <img id="image" src="graph.php" alt="">
</body>

</html>