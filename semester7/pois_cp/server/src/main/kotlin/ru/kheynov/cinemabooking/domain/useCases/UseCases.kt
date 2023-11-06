package ru.kheynov.cinemabooking.domain.useCases

import ru.kheynov.cinemabooking.domain.useCases.auth.LoginViaEmailUseCase
import ru.kheynov.cinemabooking.domain.useCases.auth.RefreshTokenUseCase
import ru.kheynov.cinemabooking.domain.useCases.auth.SignUpViaEmailUseCase

class UseCases {
    val signUpViaEmailUseCase = SignUpViaEmailUseCase()
    val loginViaEmailUseCase = LoginViaEmailUseCase()
    val refreshTokenUseCase = RefreshTokenUseCase()
    val deleteUserUseCase = DeleteUserUseCase()
    val updateUserUseCase = UpdateUserUseCase()
    val getUserDetailsUseCase = GetUserDetailsUseCase()
}