<html>

<head>
    <title>Лабораторная №2</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №2</h1>
    <h3>Вариант 12</h3>

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
            inner join workers w on w.id = d.worker
        where " . (isset($_GET['worker']) ? "lower(w.name) like lower('%" . pg_escape_string($_GET['worker']) . "%')" : "1=1")
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
            <?= isset($_SESSION['login']) ? 'Уровень доступа: ' . ($_SESSION['group'] < 2 ? 'Чтение, добавление, редактирование' : 'Чтение, добавление, удаление, редактирование') : '' ?>
        </p>
    </div>

    <form action="index.php" method="get">
        <input id="workerSearch" name="worker" />
        <label for="workerSearch">Поиск по исполнителю</label>
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