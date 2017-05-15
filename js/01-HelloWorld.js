console.log('asdf');

var i = 0;
while(i<10)
{
  console.log(i);
  i++;
}

function addMe(a, b) {
  return a + b;
}

var obj = {
  name: 'Andrew',
  age: 33
};

function Book(title, pages, isbn) {
  this.title = title;
  this.pages = pages;
  this.isbn = isbn;
}

class BookClass {
  constructor (title, pages, isbn) {
    this.title = title;
    this.pages = pages;
    this.isbn = isbn;
  }
  printIsbn() {
    console.log(this.isbn);
  }
}

class ITBook extends BookClass {
  constructor (title, pages, isbn, tech){
    super(title, pages, isbn);
    this.tech = tech;
  }
  showTech() {
      (console.log(this.tech));z
  }
}

function circleArea(r=2) {
  const PI = 3.14159
  return PI * r * r
}

squareArea = (l) => l**2;
let circleArea2 = (r) => 3.14 * r * r;

console.log(squareArea(1));

function restParameters(x, y, ...a) {
  return x + y + a.length;
}

function isEven(x) {
  return x % 2;
};

let numbers = [1, 2, 3, 4, 7, 3, 2, 7, 12];

//numbers.forEach();
