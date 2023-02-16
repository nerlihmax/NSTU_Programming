<!DOCTYPE html>
<html>

<head>
    <title>Журнал</title>
    <meta charset='utf-8'>
</head>

<body>
    <?php
    session_start();


    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["code"]) && isset($_SESSION['captcha_keystring'])) {
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

        $str = date('d.M.y h:m:s') . "\t <b>" . $_POST["username"] . "<b>\t" . $_POST["text_input"] . "\n========================\n\n";
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
        header('location:part2.php');
    }
    ?>

</body>

</html>