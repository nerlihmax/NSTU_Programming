package ru.kheynov.hotel.shared.domain.repository

import ru.kheynov.hotel.shared.domain.entities.Hotel
import ru.kheynov.hotel.shared.domain.entities.Room
import ru.kheynov.hotel.shared.domain.entities.RoomReservation
import ru.kheynov.hotel.shared.domain.entities.RoomReservationInfo
import java.time.LocalDate

interface ReservationsRepository {
    suspend fun getAvailableRooms(hotel: Hotel): List<RoomReservationInfo>
    suspend fun getHotels(): List<Hotel>
    suspend fun getRoomByID(id: String): Room?
    suspend fun getRoomOccupancy(roomId: String): List<ClosedRange<LocalDate>>
    suspend fun reserveRoom(room: RoomReservation): Boolean
    suspend fun getReservationByID(reservationId: String): RoomReservation?
    suspend fun cancelReservation(reservationId: String): Boolean
}