package ru.kheynov.hotel.domain.useCases.auth

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.shared.domain.entities.User
import ru.kheynov.hotel.shared.domain.entities.UserEmailSignUp
import ru.kheynov.hotel.shared.domain.repository.UsersRepository
import ru.kheynov.hotel.shared.jwt.RefreshToken
import ru.kheynov.hotel.shared.jwt.TokenPair
import ru.kheynov.hotel.jwt.hashing.HashingService
import ru.kheynov.hotel.jwt.token.TokenClaim
import ru.kheynov.hotel.jwt.token.TokenConfig
import ru.kheynov.hotel.jwt.token.TokenService
import java.util.UUID

class SignUpViaEmailUseCase : KoinComponent {

    private val usersRepository: UsersRepository by inject()
    private val tokenService: TokenService by inject()
    private val hashingService: HashingService by inject()
    private val tokenConfig: TokenConfig by inject()

    sealed interface Result {
        data class Successful(val tokenPair: TokenPair) : Result
        data object Failed : Result
        data object UserAlreadyExists : Result
    }

    suspend operator fun invoke(user: UserEmailSignUp): Result {
        if (usersRepository.getUserByEmail(user.email) != null) return Result.UserAlreadyExists
        val userId = getRandomUserID()
        val tokenPair = tokenService.generateTokenPair(tokenConfig, TokenClaim("userId", userId))

        val resUser = User(
            id = userId,
            name = user.name.ifEmpty { "Guest-${getRandomUsername()}" },
            email = user.email,
        )
        val registerUserResult =
            usersRepository.registerUser(
                user = resUser,
                passwordHash = hashingService.generateHash(user.password),
            )
        val createUserRefreshTokenResult = usersRepository.createRefreshToken(
            userId = userId,
            clientId = user.clientId,
            refreshToken = RefreshToken(
                token = tokenPair.refreshToken.token,
                expiresAt = tokenPair.refreshToken.expiresAt,
            ),
        )

        return if (registerUserResult && createUserRefreshTokenResult) Result.Successful(tokenPair) else Result.Failed
    }
}

fun getRandomUserID(): String = UUID.randomUUID().toString().subSequence(0..7).toString()
fun getRandomUsername(): String = UUID.randomUUID().toString().subSequence(0..6).toString()
