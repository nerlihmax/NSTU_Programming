<html>

<head>
    <title>Лабораторная №2</title>
    <style>
        #sourceLink {
            font-size: x-large;
        }
    </style>
</head>

<body>
    <a href="src.html" id="sourceLink">Исходный код</a>
    <br>
    <h1>Лабораторная №2</h1>
    <h3>Вариант 11</h3>


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
          inner join operation o on o.id = map.operation
        where " . (isset($_GET['detail']) ? "lower(map.name) like lower('%" . pg_escape_string($_GET['detail']) . "%')" : "1=1")
    );

    if (isset($_POST["log_out"])) {
        session_destroy();
        header("location:index.php");
    }

    $authenticated = !empty($_SESSION['auth']);
    $authenticated = $authenticated && $_SESSION['auth'] == true;
    ?>

    <div>
        <?php
        if ($authenticated) {
            ?>

            <form method="post" action="index.php">
                <button type="submit" name="log_out">Выход</button>
            </form>

            <?php
        } else {
            ?>
            <form method="get" action="login.php">
                <button type="submit">Вход</button>
            </form>
            <?php
        }
        ?>
        <p>
            <?= isset($_SESSION['login']) ? 'Доступные действия: ' . ($_SESSION['group'] < 2 ? 'Чтение, добавление, редактирование' : 'Чтение, добавление, удаление, редактирование') : '' ?>
        </p>
    </div>

    <form action="index.php" method="get">
        <input id="detailSearch" name="detail" />
        <label for="detailSearch">Найти по названию детали</label>
        <button type="submit">Найти</button>
    </form>

    <?php
    if ($authenticated) {
        echo '<a href="insert.php">Создать запись</a>';
    }
    ?>
    <br>
    <br>
    <table border="1">
        <tr>
            <th>ID</th>
            <th>Название детали</th>
            <th>Вид операции</th>
            <th>Длительность операции в минутах</th>

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
        while ($line = pg_fetch_array($detail, null, PGSQL_ASSOC)) {

            echo "\t<tr>\n";
            foreach ($line as $col_value) {
                ?>
                <td>
                    <?= $col_value ?>
                </td>
                <?php
            }
            if ($authenticated && $_SESSION['group'] >= 1) {
                echo "\t<td><a href=\"update.php?id=" . $line["id"] . "\">Изменить</a></td>";
                if ($authenticated && $_SESSION['group'] >= 2) {
                    echo "\t<td><a href=\"delete.php?id=" . $line["id"] . "\">Удалить</a></td>";
                }
            }
        }
        ?>
        </tr>
    </table>

</body>

</html>