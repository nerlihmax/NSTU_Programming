package ru.kheynov.hotel.utils

import android.content.Context
import androidx.annotation.StringRes
import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource

/**
 * [UiText] is a wrapper for text to decouple it from UI and [Context]
 */
sealed class UiText {
    data class PlainText(val value: String) : UiText()
    class StringResource(@StringRes val resId: Int) : UiText()

    @Composable
    fun asString(): String {
        return when (this) {
            is PlainText -> value
            is StringResource -> stringResource(resId)
        }
    }
}
