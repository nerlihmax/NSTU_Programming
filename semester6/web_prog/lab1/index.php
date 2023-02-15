<html>

<head>
    <title>Web programming NSTU course</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №1</h1>
    <h3>Вариант 17</h3>

    <h2>Часть 1</h2>
    <a href="dump.sql">
        <h3><b>SQL дамп:<b></h3><br>
        <img id="dlimg" src="img.png" height="100px" />
    </a>

    <br>
    <br>
    <h2>Часть 2</h2>



    <h3>Теневая посылка</h3>

    <?php
    ini_set('display_errors', '1');
    ini_set('display_startup_errors', '1');
    error_reporting(E_ALL);


    if (isset($_COOKIE["visit_date"]) && isset($_COOKIE["message"])) {
        echo $_COOKIE["visit_date"] . "<br>" . "<span>Приветствие: " . $_COOKIE["message"] . "</span>";
    }
    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["message"])) {
        setcookie("visit_date", date('d.M.y h:m:s'), time() + 365 * 24 * 60 * 60);
        setcookie("message", $_POST["message"], time() + 365 * 24 * 60 * 60);
        header("location:index.php");
    }


    ?>

    <form method="post" action="index.php">
        <P><span>Приветствие: </span><br><input type="text" name="message"></P>
        <P><button type="submit">Отправить</button></P>
    </form>

    <h3>Текстовый файл</h3>

    <form method="post" action="index.php">
        <P><span>Имя туриста: </span><br><input type="text" name="tourist_name"></P>
        <P><span>Пожелания по формам отдыха: </span><br><textarea type="text" name="wishes"></textarea></P>
        <P><button type="submit">Отправить</button></P>
    </form>

    <?php

    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["wishes"]) && isset($_POST["tourist_name"])) {
        $fp = @fopen(__DIR__ . "/log.txt", "a");
        if (!$fp) {
            echo "Ваша анкета не может быть обработана сейчас!";
            exit;
        }

        $str = date('d.M.y h:m:s') . "\t" . $_POST["tourist_name"] . "\t" . $_POST["wishes"] . "\n";
        fwrite($fp, $str);
        header("location:index.php");
    }

    if (!file_exists(__DIR__ . "/log.txt")) {
        echo "Файл журнала не найден";
        exit;
    }
    echo "<h3>Журнал</h3>";
    $file = fopen(__DIR__ . '/log.txt', 'r');
    while (!feof($file)) {
        $endLine = fgets($file);
        echo $endLine . "<br>";
    }

    fclose($file);
    ?>


</body>

</html>