


#include "RandBLAS/config.h"
#include "RandBLAS/exceptions.hh"

#include <string>
#include <gtest/gtest.h>

class TestExceptions : public ::testing::Test {
    protected:
};

TEST_F(TestExceptions, randblas_require_var_arg) {
    bool successful_raise = false;
    try {
        randblas_require(successful_raise);
    } catch (RandBLAS::exceptions::Error &e) {
        std::string message{e.what()};
        successful_raise = message.find("successful_raise") != std::string::npos;
    }
    ASSERT_TRUE(successful_raise);
}

TEST_F(TestExceptions, randblas_require_expr_arg) {
    int flag = 0;
    try {
        randblas_require(flag > 1);
    } catch (RandBLAS::exceptions::Error &e) {
        std::string message{e.what()};
        flag = (int) message.find("flag > 1") != std::string::npos;
    }
    ASSERT_TRUE(flag);
}

TEST_F(TestExceptions, randblas_error_if_msg_output) {
    bool error_trigger = true;
    bool expect_true = false;
    try {
        randblas_error_if_msg(error_trigger, "Custom message.");
    } catch (RandBLAS::exceptions::Error &e) {
        std::string message{e.what()};
        expect_true = message.find("Custom message.") != std::string::npos;
    }
    ASSERT_TRUE(expect_true);
}
