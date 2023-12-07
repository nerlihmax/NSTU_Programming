package ru.kheynov.hotel.domain.repository

import ru.kheynov.hotel.domain.entities.Hotel
import ru.kheynov.hotel.domain.entities.Reservation
import ru.kheynov.hotel.domain.entities.RoomReservation

interface ReservationsRepository {
    suspend fun getAvailableRooms(hotel: Hotel): List<RoomReservation>
    suspend fun getHotels(): List<Hotel>
    suspend fun reserveRoom(room: RoomReservation): Boolean
    suspend fun cancelReservation(reservationId: String): Boolean
}