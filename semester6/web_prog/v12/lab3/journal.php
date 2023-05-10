<!DOCTYPE html>
<html>

<head>
    <title>Журнал</title>
    <meta charset='utf-8'>
</head>

<body>
    <?php
    ini_set('display_errors', '1');
    ini_set('display_startup_errors', '1');
    error_reporting(E_ALL);

    session_start();

    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["styles"]) && isset($_POST["size"]) && isset($_SESSION['captcha_keystring'])) {
        if ($_SESSION['captcha_keystring'] != $_POST["code"]) {
            header('location:part2.php');
            exit;
        }
        unset($_SESSION['captcha_keystring']);

        $fp = @fopen(__DIR__ . "/log.txt", "a");
        if (!$fp) {
            echo "Ваша анкета не может быть обработана сейчас!";
            exit;
        }

        $style_checked = $_POST['styles'];
        $style = '';
        for ($i = 0; $i < count($style_checked); $i++) {
            $style = $style . $style_checked[$i] . "\t";
        }

        $str = "Виды приветствия: " . $style .
            "\nРазмер сообщения: " . $_POST["size"] .
            "\nКомментарий: " . $_POST["text_input"] . "\n";
        fwrite($fp, $str);

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
    } else {
        session_abort();
        header('location:part2.php');
    }
    ?>

</body>

</html>