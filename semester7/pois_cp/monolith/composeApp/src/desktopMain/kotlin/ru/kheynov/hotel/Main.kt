package ru.kheynov.hotel

import androidx.compose.foundation.layout.width
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import org.koin.core.context.startKoin
import ru.kheynov.hotel.desktop.di.appModule
import ru.kheynov.hotel.desktop.presentation.HomePage

fun main() = application {
    startKoin {
        modules(appModule)
    }
    Window(
        onCloseRequest = ::exitApplication,
        state = rememberWindowState(width = 1200.dp, height = 1000.dp),
        title = "Система автоматизации резервирования номеров в гостинице",
    ) {
        App()
    }
}

@Composable
fun App() {
    MaterialTheme {
        Surface(modifier = Modifier.width(5000.dp)) {
            HomePage()
        }
    }
}

