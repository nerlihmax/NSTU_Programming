package ru.kheynov.hotel.shared.data.repository

import ru.kheynov.hotel.shared.data.api.ReservationsAPI
import ru.kheynov.hotel.shared.data.models.reservations.ReserveRoomRequest
import ru.kheynov.hotel.shared.domain.entities.Hotel
import ru.kheynov.hotel.shared.domain.entities.RoomInfo
import java.time.LocalDate

class ReservationsRepository(
    private val api: ReservationsAPI
) {
    suspend fun getReservations() = api.getReservations()

    suspend fun getReservations(userId: String) = api.getReservations(userId)

    suspend fun addReservation(
        roomId: String,
        from: String,
        to: String,
    ) = api.addReservation(ReserveRoomRequest(roomId, from, to))

    suspend fun deleteReservation(id: String): String = api.deleteReservation(id)

    suspend fun getHotels(): List<Hotel> = api.getHotels()

    suspend fun getRooms(hotelId: Int): List<RoomInfo> = api.getRooms(hotelId)

    suspend fun getRoomsOccupancy(roomId: String): List<ClosedRange<LocalDate>> =
        api.getRoomsOccupancy(roomId)
}