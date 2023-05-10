<html>
<style>
    .block__download {
        font-size: 24px;
        padding: 4px;

        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }

    .block__cookies {
        padding: 8px;

        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .block__text {
        padding: 8px;

    }

    .form {
        padding: 2px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .form__flags {
        display: flex;
        gap: 4px;
        align-items: center;
    }

    .form__flags__div {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
    }

    .journal {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
    }

    #dump {
        font-size: xx-large;
    }
</style>

<head>
    <title>Web - lab4</title>
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.3/dist/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-validate/1.19.0/jquery.validate.min.js"></script>
    <link rel="stylesheet" type="text/css" href="../style.css">
</head>

<body>
    <header class="header">
        <h1 class="h1">Лабораторная работа №4, часть 1</h1>
    </header>

    <button onclick="window.location.href = 'http://217.71.129.139:5473/lab4/sources.html'" class="dump">Исходный
        код</button>

    <div class="main__div">
        <main class="main">

            <h3 class="h2">Валидация JS</h3>

            <section class="block__cookies">
                <?php
                $flags = "отсутствует";

                function flagsName($name)
                {
                    if ($name == "text") {
                        return "Текст";
                    } else if ($name == "graphics") {
                        return "Графика";
                    } else if ($name == "styles") {
                        return "Стили";
                    }

                    return "отсутствует";
                }

                if ($_SERVER['REQUEST_METHOD'] == 'POST' && $_POST["comment"] == "" && $_POST["styles"] == [] && $_POST["flags"] != [] || ($_POST["flags"] != [] && $_POST["comment"] != "" && $_POST["styles"] != [])) {
                    setcookie("visit", date('m/d/y h:m'), time() + 365 * 24 * 60 * 60);

                    if (isset($_POST["flags"])) {
                        setcookie("flags", json_encode($_POST["flags"]), time() + 365 * 24 * 60 * 60);
                    } else {
                        setcookie("flags", json_encode([""]), time() + 365 * 24 * 60 * 60);
                    }

                    header("location:lab7.php");
                }

                if (isset($_COOKIE["visit"]) && isset($_COOKIE["flags"])) {
                    $text = json_decode($_COOKIE["flags"]);
                    $flags = "";

                    for ($i = 0; $i < count($text); $i++) {
                        $flags = $flags . flagsName($text[$i]);
                        if ($i != count($text) - 1) {
                            $flags = $flags . "---";
                        }
                    }

                    echo "<p>Время последнего сохранение данных: " . $_COOKIE["visit"] . "</p>";
                }
                ?>
                <p>Ваш выбор:
                    <?= $flags ?>
                </p>

                <form class="form" id="form1" method="post" action="lab7.php">
                    <div class="form__flags">
                        <p>Набор флажков:</p>
                        <div class="form__flags__div">
                            <p><input type="checkbox" id="text" value="text" name="flags[]"></p>
                            <label for="text">Текст</label>
                        </div>
                        <div class="form__flags__div">
                            <p><input type="checkbox" id="graphics" value="graphics" name="flags[]"></p>
                            <label for="graphics">Графика</label>
                        </div>
                        <div class="form__flags__div">
                            <p><input type="checkbox" id="styles" value="styles" name="flags[]"></p>
                            <label for="styles">Стили</label>
                        </div>
                    </div>

                    <div>
                        <input type="submit" value="Отправить">
                    </div>
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
                        const isCheckboxesValid = checkboxes("flags[]")
                        if (!isCheckboxesValid) {
                            event.preventDefault()
                            const res = "Стиль оформления не выбран"
                            alert(res)
                        }
                    })
                </script>
            </section>
            <section>
                <h3 class="h2">Валидация Jquery </h3>

                <form class="form" id="form2" method="post" action="lab7.php">
                    <div class="form__flags">
                        <p>Оформления текста:</p>
                        <div class="form__flags__div">
                            <p><input TYPE="checkbox" id="bold" value="bold" name="styles[]"></p>
                            <label for="cat">Жирный</label>
                        </div>
                        <div class="form__flags__div">
                            <p><input TYPE="checkbox" id="italic" value="italic" name="styles[]"></p>
                            <label for="dog">Курсив</label>
                        </div>
                        <div class="form__flags__div">
                            <p><input TYPE="checkbox" id="dotted" value="dotted" name="styles[]"></p>
                            <label for="turtle">В горошек</label>
                        </div>
                    </div>

                    <div>
                        <p>Ваш комментарий: </p>
                        <p><textarea rows="4" cols="50" type="text" name="comment"></textarea></p>
                    </div>

                    <div>
                        <input type="submit" value="Отправить">
                    </div>
                </form>
                <script>
                    $(document).ready(function () {
                        $("#form2").validate({
                            rules: {
                                comment: {
                                    required: true,
                                    minlength: 4
                                },
                                'styles[]': {
                                    required: true,
                                    minlength: 1
                                },
                            },
                            messages: {
                                comment: {
                                    required: "Комментарий не может быть пустым. ",
                                    minlength: "Комментарий должен быть длиннее 4 символов. "
                                },
                                'styles[]': {
                                    required: "Стиль оформления не выбран. ",
                                    minlength: "Стиль оформления не выбран. "
                                },
                            },
                            errorElement: "div",
                            // errorClass: '.error',
                            errorPlacement: function (error, element) {
                                if (element.is(":checkbox")) {
                                    error.appendTo(element.parents('.form__flags'));
                                }
                                else {
                                    error.insertAfter(element);
                                }
                            },
                        })
                    });
                </script>
            </section>
            <section>
                <?php
                $styles = "";
                $comment = "---";

                function stylesName($name)
                {
                    if ($name == "bold") {
                        return "Жирный";
                    } else if ($name == "italic") {
                        return "Курсив";
                    } else if ($name == "dotted") {
                        return "В горошек";
                    }

                    return "---";
                }

                if ($_SERVER['REQUEST_METHOD'] == 'POST' && $_POST["flags"] == [] && ($_POST["comment"] != "" || $_POST["styles"] != []) || ($_POST["flags"] != [] && $_POST["comment"] != "" && $_POST["styles"] != [])) {
                    if (isset($_POST["styles"])) {
                        for ($i = 0; $i < count($_POST["styles"]); $i++) {
                            $styles = $styles . stylesName($_POST["styles"][$i]);
                            if ($i != count($_POST["styles"]) - 1) {
                                $styles = $styles . " | ";
                            }
                        }
                    } else {
                        $styles = "---";
                    }

                    if (isset($_POST["comment"]) && $_POST["comment"] != "") {
                        $comment = $_POST["comment"];
                    }

                    $fp = @fopen(__DIR__ . "/logs.txt", "a");
                    if (!$fp) {
                        echo "<p>Ошибка при открытии файла</p>";
                    } else {
                        $str = "<p>" . date('d/m/y h:m') . "</p>" . "<p>" . $styles . "</p>" . "<p>" . $comment . "</p>";
                        fwrite($fp, $str);
                    }

                    header("location:lab7.php");
                }

                if (!file_exists(__DIR__ . "/logs.txt")) {
                    echo "<p>Ошибка при открытии журнала</p>";
                } else {
                    echo '<h3 class="h2">Журнал</h3>';
                    echo '<div class="journal"><p>Дата</p><p>Стили</p><p>Комментарий</p>';

                    $file = fopen(__DIR__ . '/logs.txt', 'r');
                    while (!feof($file)) {
                        $text = fgets($file);
                        echo $text;
                    }

                    fclose($file);
                }
                ?>
            </section>
        </main>
    </div>
</body>

</html>