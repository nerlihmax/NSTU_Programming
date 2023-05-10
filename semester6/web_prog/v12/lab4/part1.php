<?php
define('LOG_PATH', __DIR__ . '/log.txt');

function write_log($style, $size)
{
    $line = [
        'style' => $style,
        'size' => $size,
    ];
    $json = json_encode($line);
    file_put_contents(LOG_PATH, "$json\n", FILE_APPEND);
}

function read_log()
{
    $log = [];
    $json_lines = file_get_contents(LOG_PATH);

    foreach (explode("\n", $json_lines) as $line) {
        if (empty($line))
            continue;
        $log[] = json_decode($line, true);
    }

    return $log;
}

$secs_in_year = time() + 60 * 60 * 24 * 365;

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["text_styles"])) {
    $checked = $_POST['text_styles'];
    $result = '';
    for ($i = 0; $i < count($checked); $i++) {
        $result = $result . $checked[$i] . "\t";
    }
    setcookie("text_styles", $result, $secs_in_year);
    header("location:part1.php");
}

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["styles"]) && isset($_POST["size"])) {
    $style_checked = $_POST['styles'];
    for ($i = 0; $i < count($style_checked); $i++) {
        $style = $style . $style_checked[$i] . "\t";
    }
    write_log($style, $_POST['size']);
    header('location:part1.php');
}
?>

<html>

<head>
    <title>Лабораторная №4</title>
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.3/dist/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-validate/1.19.0/jquery.validate.min.js"></script>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }

        .error {
            color: red;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №4</h1>
    <h3>Вариант 12</h3>

    <div>
        <h2>Валидация JavaScript</h2>

        <?php
        if (isset($_COOKIE["text_styles"])) {
            echo "<p>Выбранные стили текста: " . $_COOKIE["text_styles"] . "</p>";
        }
        ?>

        <h2>Выберите стили текста: </h2>
        <form id="form1" method="post" action="part1.php">
            <div>
                <input type="checkbox" name="text_styles[]" value="bold" /> Жирный<br />
                <input type="checkbox" name="text_styles[]" value="semibold" /> Полужирный<br />
                <input type="checkbox" name="text_styles[]" value="italic" /> Курсив<br />
            </div>
            <br>
            <button s type="submit">Отправить</button>
        </form>
        <script>
            function checkboxes(name) {
                var checkboxes = document.getElementsByName(name);
                var formValid = false;

                var i = 0;
                while (!formValid && i < checkboxes.length) {
                    if (checkboxes[i].checked) formValid = true;
                    i++;
                }
                return formValid;
            }

            $("#form1").on("submit", (event) => {
                const isCheckboxesValid = checkboxes("text_styles[]")
                if (!isCheckboxesValid) {
                    event.preventDefault()
                    const res = "Стиль приветствия не выбран"
                    alert(res)
                }
            })
        </script>
    </div>

    <div>
        <h2>Валидация JQuery</h2>
        <div>
            <form id="form2" method="post" action="part1.php">
                <h3>Выберите стили приветствия:</h3>
                <div id="styles_checkboxes">
                    <input type="checkbox" name="styles[]" value="official" /> Официальное<br />
                    <input type="checkbox" name="styles[]" value="nonformal" /> Неофициальное<br />
                    <input type="checkbox" name="styles[]" value="basic" /> Простое<br />
                </div>

                <h3>Выберите размер: </h3>
                <select name="size">
                    <option value="" selected disabled hidden>-- Размер --</option>
                    <option value="10 pt">Маленький</option>
                    <option value="18 pt">Средний</option>
                    <option value="32 pt">Большой</option>
                </select>
                <br>
                <br>
                <button type="submit">Отправить</button>
            </form>
            <script>
                $(document).ready(function () {
                    $("#form2").validate({
                        rules: {
                            'styles[]': {
                                required: true,
                                minlength: 1
                            },
                            size: {
                                required: true,
                            }
                        },
                        messages: {
                            'styles[]': {
                                required: "Стили приветствия не выбраны.",
                                minlength: "Стили приветствия не выбраны."
                            },
                            size: {
                                required: "Размер не выбран. ",
                            }
                        },
                        errorElement: "div",
                        errorPlacement: function (error, element) {
                            if (element.is(":checkbox")) {
                                error.appendTo(element.parents('#styles_checkboxes'));
                            }
                            else {
                                error.insertAfter(element);
                            }
                        },
                    })
                });
            </script>
        </div>
    </div>

    <div>
        <h2>Журнал</h2>
        <?php
        $log = read_log();
        if (!empty($log)) {
            foreach ($log as $element) {
                $style = $element['style'];
                $size = $element['size'];
                ?>
                <div>
                    <p>
                        Вид приветствия:
                        <?= htmlspecialchars($style) ?>
                    </p>
                    <p>
                        Размер:
                        <?= htmlspecialchars($size) ?>
                    </p>
                </div>
                <?php
            }
        }
        ?>
    </div>
</body>

</html>