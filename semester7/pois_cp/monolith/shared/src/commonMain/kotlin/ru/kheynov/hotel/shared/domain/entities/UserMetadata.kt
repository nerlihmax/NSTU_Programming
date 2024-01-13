package ru.kheynov.hotel.shared.domain.entities

sealed interface UserMetadataState {
    data object Loading : UserMetadataState
    data class UserMetadata(
        val name: String,
        val isEmployee: Boolean,
        val hotel: Hotel,
    ) : UserMetadataState
}
