package ru.kheynov.hotel.desktop.presentation.stateHolders

sealed interface Routes {
    data object Rooms : Routes
    data object Bookings : Routes
}