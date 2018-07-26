#include <iostream>


inline bool sum_of_two_ints(float a, float b) { return a + b > 5; }

void add_three(int &x)
{
    x = x + 3;
}

int main()
{
    int j = 8;
    int i = 4;
    int *pointer_to_i = &i;
    int *pointer_to_nothing;
    //int &thing_to_i = 5;
    std::cout << "i: " << i << std::endl;
    std::cout << " &i: " << &i << std::endl;
    std::cout << "*pointer_to_i: " << *pointer_to_i << std::endl;
    std::cout << "&*poijter_to_i: " << &*pointer_to_i << std::endl;
    std::cout << "pointer_to_i: " << pointer_to_i << std::endl;
    std::cout << "&pointer_to_i: " << &pointer_to_i << std::endl;
    int number = 88;
    int *pNumber = &number;
    int *pAnotherOfSame = &number;
    std::cout << number << std::endl;
    std::cout << *pNumber << std::endl;
    std::cout << typeid(number).name() << std::endl;
    *pNumber = number + 12;
    std::cout << number << std::endl;
    std::cout << "&number: " << &number << std::endl;
    int *pNumber_plus_2 = pNumber + 2;
    int *pNumber_plus_deux = &number + 2;
    std::cout << "pNumber: " << pNumber << std::endl;
    std::cout << "pNumber_plus_2: " << pNumber_plus_2 << std::endl;
    std::cout << "*pNumber_plus_2: " << *pNumber_plus_2 << std::endl;
    std::cout << "*pNumber_plus_deax: " << *pNumber_plus_deux << std::endl;
    int * pointer_new = new int;
    std::cout << pointer_new << std::endl;
    std::cout << *pointer_new << std::endl;
    std::cout << sum_of_two_ints(2.2, 5) << std::endl;
    delete pointer_new;
    int k = 6;
    std::cout << "k: " << k << std::endl;
    // std::cout << "add three(k): " << add_three(k) << std::endl;
    add_three(k);
    std::cout << "k: " << k << std::endl;

    return 0;

}
