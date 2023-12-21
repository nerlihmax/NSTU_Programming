package ru.kheynov.hotel.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material.Colors
import androidx.compose.material.MaterialTheme
import androidx.compose.material.darkColors
import androidx.compose.material.lightColors
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color


private val DarkColorPalette = darkColors(
    primary = colorPrimaryDark,
    primaryVariant = colorPrimaryDark,
    secondary = colorAccentDark,
    background = colorBackgroundPrimaryDark,
    surface = colorBackgroundSecondaryDark,
)

private val LightColorPalette = lightColors(
    primary = colorPrimaryLight,
    primaryVariant = colorPrimaryLight,
    secondary = colorAccentLight,
    background = colorBackgroundPrimaryLight,
    surface = colorBackgroundSecondaryLight,
)

val Colors.red: Color get() = if (isLight) redLight else redDark
val Colors.green: Color get() = if (isLight) greenLight else greenDark
val Colors.blue: Color get() = if (isLight) blueLight else blueDark
val Colors.lightBlue: Color get() = if (isLight) lightBlueLight else lightBlueDark
val Colors.separator: Color get() = if (isLight) separatorLight else separatorDark
val Colors.gray: Color get() = if (isLight) grayLightPalette else grayDark
val Colors.grayLight: Color get() = if (isLight) grayLightLight else grayLightDark
val Colors.overlay: Color get() = if (isLight) overlayLight else overlayDark
val Colors.disabled: Color get() = if (isLight) colorDisableLight else colorDisableDark
val Colors.tertiary: Color get() = if (isLight) colorTertiaryLight else colorTertiaryDark
val Colors.elevated: Color get() = if (isLight) colorBackgroundElevatedLight else colorBackgroundElevatedDark


@Composable
fun AppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colors = if (darkTheme) {
        DarkColorPalette
    } else {
        LightColorPalette
    }

    MaterialTheme(
        colors = colors,
        typography = Typography,
        shapes = Shapes,
        content = content,
    )
}