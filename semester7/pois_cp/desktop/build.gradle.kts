import org.jetbrains.compose.compose
import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("multiplatform")
    id("org.jetbrains.compose")
//    id("org.jlleitschuh.gradle.ktlint") version "11.3.1"
}

//ktlint {
//    ignoreFailures.set(false)
//    @Suppress("DEPRECATION")
//    disabledRules.set(setOf("no-wildcard-imports"))
//    reporters {
//        reporter(org.jlleitschuh.gradle.ktlint.reporter.ReporterType.PLAIN)
//        reporter(org.jlleitschuh.gradle.ktlint.reporter.ReporterType.CHECKSTYLE)
//        reporter(org.jlleitschuh.gradle.ktlint.reporter.ReporterType.SARIF)
//    }
//}

group = "ru.kheynov"
version = "1.0-SNAPSHOT"

repositories {
    google()
    mavenCentral()
    maven("https://maven.pkg.jetbrains.space/public/p/compose/dev")
}

kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "11"
        }
        withJava()
    }
    sourceSets {
        val jvmMain by getting {
            dependencies {
                implementation(compose.desktop.currentOs)

                // Database
                implementation("org.ktorm:ktorm-core:3.5.0")
                implementation("org.ktorm:ktorm-support-postgresql:3.5.0")
                implementation("org.postgresql:postgresql:42.5.1")
            }
        }
        val jvmTest by getting
    }
}

compose.desktop {
    application {
        mainClass = "MainKt"
        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "rgr"
            packageVersion = "1.0.0"
        }
    }
}
