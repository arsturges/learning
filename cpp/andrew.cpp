#include <iostream>
using std::cin;
using std::cout;
using std::endl;
int main()
{
    cout << "Green rhmyes with 13." << endl;
    int signed_dogs = INT_MAX;
    unsigned int unsigned_dogs= signed_dogs;
    cout << "You'll find this to be true." << endl;
    cout << "SHRT_MAX: " << SHRT_MAX << ", sizeof =  " << sizeof(SHRT_MAX) << endl;
    cout << "LONG_MAX: " << LONG_MAX << ", sizeof =  " << sizeof(LONG_MAX) << endl;
    cout << "LLONG_MAX: " << LLONG_MAX << ", sizeof =  " << sizeof(LLONG_MAX) << endl;
    cout << "LLONG_MIN: " << LLONG_MIN << ", sizeof =  " << sizeof(LLONG_MIN) << endl;
    cout << "signed_dogs: " << signed_dogs + 1 << endl;
    cout << "unsigned_dogs: " << unsigned_dogs +1  << endl;


    cout << "Enter a character: " << endl;
    char character = 'a';
    cout << "Hola! Thank you for the  " << character << " character!" << endl;
    //cout.put("&")
    int character_digit = character;
    cout.put(character_digit);
    cout << "char digit: " << character_digit << endl;
    float flights_per_day = 12.0002;
    cout<< "flights: " << flights_per_day << endl;

    int cards[4] = {3, 6, 8, 10};

    for (int i = 0; i < 4; i++)
    {
        cout << i << endl;
    }
    cout << sizeof(cards) << endl;
    return 0;
}
