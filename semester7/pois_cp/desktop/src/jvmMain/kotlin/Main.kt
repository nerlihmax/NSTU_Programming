import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.layout.width
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import presentation.MainPageV7

@Composable
@Preview
fun App() {
    MaterialTheme {
        Surface(modifier = Modifier.width(5000.dp)) {
            MainPageV7()
        }
    }
}

fun main() = application {
    Window(
        onCloseRequest = ::exitApplication,
        state = rememberWindowState(width = 1200.dp, height = 1000.dp),
        title = "Расчётно-графическая работа вариант 7",
    ) {
        App()
    }
}
