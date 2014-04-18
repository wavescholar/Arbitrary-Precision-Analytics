// ArbitraryPrecisionRootFinder.cpp 
//Bruce B Campbell for Dr. Mari Mori
//#define BOOST_MATH_INSTRUMENT
#include <boost/math/tools/roots.hpp>
#include <limits>

namespace boost{ namespace math{ namespace tools{

	template <class F, class T>
	T newton_raphson_iterate(F f, T guess, T min, T max, int digits);

	template <class F, class T>
	T newton_raphson_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);

	template <class F, class T>
	T halley_iterate(F f, T guess, T min, T max, int digits);

	template <class F, class T>
	T halley_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);

	template <class F, class T>
	T schroeder_iterate(F f, T guess, T min, T max, int digits);

	template <class F, class T>
	T schroeder_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);

}}} 

/*
These functions all perform iterative root finding using derivatives: 

newton_raphson_iterateperforms second order Newton-Raphson iteration, 
halley_iterate andschroeder_iterate perform third order Halley and Schroeder iteration. 
The functions all take the same parameters: 

Parameters of the root finding functions

F f 
Type F must be a callable function object that accepts one parameter and returns a boost::math::tuple: 

For the second order iterative methods (Newton Raphson) the boost::math::tuple should have two elements containing the evaluation of the function and its first derivative. 

For the third order methods (Halley and Schroeder) the boost::math::tuple should have three elements containing the evaluation of the function and its first and second derivatives. 

T guess 
The initial starting value. A good guess is crucial to quick convergence! 

T min 
The minimum possible value for the result, this is used as an initial lower bracket. 

T max 
The maximum possible value for the result, this is used as an initial upper bracket. 

int digits 
The desired number of binary digits. 

uintmax_t max_iter 
An optional maximum number of iterations to perform. 

When using these functions you should note that: 

Default max_iter = (std::numeric_limits<boost::uintmax_t>::max)() is effectively 'iterate for ever'!. 
They may be very sensitive to the initial guess, typically they converge very rapidly if the initial guess has 
two or three decimal digits correct. However convergence can be no better than bisection, or in some rare cases, 
even worse than bisection if the initial guess is a long way from the correct value and the derivatives are close to zero. 
These functions include special cases to handle zero first (and second where appropriate) derivatives, and fall back to 
bisection in this case. However, it is helpful if functor F is defined to return an arbitrarily small value of the correct
sign rather than zero. 
If the derivative at the current best guess for the result is infinite (or very close to being infinite) then these functions 
may terminate prematurely. A large first derivative leads to a very small next step, triggering the termination condition. 
Derivative based iteration may not be appropriate in such cases. 
If the function is 'Really Well Behaved' (monotonic and has only one root) the bracket bounds min and max may as well be set
to the widest limits like zero and numeric_limits<T>::max(). 
But if the function more complex and may have more than one root or a pole, the choice of bounds is protection against 
jumping out to seek the 'wrong' root. 
These functions fall back to bisection if the next computed step would take the next value out of bounds. The bounds are
updated after each step to ensure this leads to convergence. However, a good initial guess backed up by asymptotically-tight
bounds will improve performance no end - rather than relying on bisection. 
The value of digits is crucial to good performance of these functions, if it is set too high then at best you will get one
extra (unnecessary) iteration, and at worst the last few steps will proceed by bisection. Remember that the returned value can
never be more accurate than f(x) can be evaluated, and that if f(x) suffers from cancellation errors as it tends to zero then
the computed steps will be effectively random. The value of digits should be set so that iteration terminates before this 
point: remember that for second and third order methods the number of correct digits in the result is increasing quite 
substantially with each iteration, digits should be set by experiment so that the final iteration just takes the next 
value into the zone where f(x) becomes inaccurate. 
To get the binary digits of accuracy, use policies::get_max_root_iterations<Policy>()). 
If you need some diagnostic output to see what is going on, you can #define BOOST_MATH_INSTRUMENT before the 
#include <boost/math/tools/roots.hpp>, and also ensure that display of all the possibly significant digits with 
cout.precision(std::numeric_limits<double>::max_digits10): but be warned, this may produce copious output! 
Finally: you may well be able to do better than these functions by hand-coding the heuristics used so that they 
are tailored to a specific function. You may also be able to compute the ratio of derivatives used by these methods 
more efficiently than computing the derivatives themselves. As ever, algebraic simplification can be a big win. 

Newton Raphson Method 
Given an initial guess x0 the subsequent values are computed using: 
x_{n+1} = x{n}- \frac{f(x)}{f'(x)}

Out of bounds steps revert to bisection of the current bounds. 

Under ideal conditions, the number of correct digits doubles with each iteration. 

Halley's Method 
Given an initial guess x0 the subsequent values are computed using: 
x_{n+1} = x{n}- \frac{2f(x)*f'(x)}{2(f'(x))^2 - f(x)*f''(x)} 

Over-compensation by the second derivative (one which would proceed in the wrong direction) causes the method to revert to a Newton-Raphson step. 

Out of bounds steps revert to bisection of the current bounds. 

Under ideal conditions, the number of correct digits trebles with each iteration. 

Schroeder's Method 
Given an initial guess x0 the subsequent values are computed using: 
x_{n+1} = x{n}- \frac{f(x)}{f'(x)} - \frac{f''(x)(f(x))^2}{2(f'(x))^3}

Over-compensation by the second derivative (one which would proceed in the wrong direction) causes the method to revert to a Newton-Raphson step. Likewise a Newton step is used whenever that Newton step would change the next value by more than 10%. 

Out of bounds steps revert to bisection of the current bounds. 

Under ideal conditions, the number of correct digits trebles with each iteration. 

Example 
Let's suppose we want to find the cube root of a number: the equation we want to solve along with its derivatives are: 
f(x) = x^3-a
f'(x) = 3x^2;
f''(x)=6x;


To begin with lets solve the problem using Newton-Raphson iterations, we'll begin by defining a function object 
(functor) that returns the evaluation of the function to solve, along with its first derivative f'(x): 
The code below is for Newton-Rhaphson

template <class T>
struct cbrt_functor
{
cbrt_functor(T const& target) : a(target)
{ // Constructor stores value to be 'cube-rooted'.
}
boost::math::tuple<T, T> operator()(T const& z)
{ // z is estimate so far.
return boost::math::make_tuple(
z*z*z - a, // return both f(x)
3 * z*z);  // and f'(x)
}
private:
T a; // to be 'cube-rooted'.
};

template <class T>T cbrt(T z){
using namespace std; // for frexp, ldexp, numeric_limits.
using namespace boost::math::tools;   int exp;
frexp(z, &exp); // Get exponent of z (ignore mantissa).
T min = ldexp(0.5, exp/3);   T max = ldexp(2.0, exp/3);
T guess = ldexp(1.0, exp/3); // Rough guess is to divide the exponent by three.
int digits = std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy for type T.
return newton_raphson_iterate(detail::cbrt_functor<T>(z), guess, min, max, digits);
}

The implementation below in this header uses Halley's method.

*/

using namespace boost; 
template <class T>

struct cbrt_functor
{
	cbrt_functor(T const& target) : a(target){}
	std::tr1::tuple<T, T, T> operator()(T const& z)
	{
		T sqr = z * z;
		return std::tr1::make_tuple(sqr * z - a, 3 * sqr, 6 * z);
	}
private:
	T a;
};

template <class T>
T kl_arbitrary_precision_root(T z, unsigned int precision)
{
	using namespace std;
	int exp;
	frexp(z, &exp);
	T min = ldexp(0.5, exp/3);
	T max = ldexp(2.0, exp/3);
	T guess = ldexp(1.0, exp/3);
	int digits = std::numeric_limits<T>::digits / 2;

	//cout<<"Typeid for  call to kl_arbitrary_precision_root(T z, unsigned int precision) "<<typeid(z).name()<<endl<<endl;

	string zType =typeid(z).name();
	int usingArb = zType.compare(typeid(boost::math::ntl::RR).name());
	if (usingArb==0)
	{
		//cout<<"We're using arbitrary precision in kl_arbitrary_precision_root"<<endl<<endl;;
		digits = precision /2;
	}

	T ans =boost::math::tools::halley_iterate(cbrt_functor<T>(z), guess, min, max, digits);
	return ans;
}

#include <iostream>
using namespace std;

#include <float.h>

/*
Calculates inverse cube root value. Vector variant of invcbrt(x) function for a 128-bit/256-bit vector argument of float64 values.
*/
//#include <mathimf.h >
//__m128d _mm_invcbrt_pd(__m128d v1);
//double cbrt(double x);
//long double cbrtl(long double x);
//float cbrtf(float x);

//For NTL
#include < boost/math/bindings/rr.hpp>
using boost::math::ntl::RR;


//For QueryPerformanceCounter
#include <tchar.h>
#include <windows.h>

int testMaxCampbellPrime();

int main(int argc, char* argv[])
{
	testMaxCampbellPrime();
	cout<<endl<<endl<<"Put an integer number in  - like 794"<<endl<<endl;
	double mariIn ;
	cin>>mariIn;
	do
	{
		cout<<"Ok Then we're going to fnd the root of "<<mariIn<<endl;
		cout<<"First we're going to clear the floating point error registers on the microprocessor."<<endl<<endl;

		/*
		First a bit about Floating Point Exceptions

		The IEEE floating point standard defines several exceptions that occur when the result of a floating point operation is unclear or undesirable. 
		Exceptions can be ignored, in which case some default action is taken, such as returning a special value. 
		When trapping is enabled for an exception, a error is signalled whenever that exception occurs. These are the possible floating point exceptions: 

		:underflow 
		This exception occurs when the result of an operation is too small to be represented as a normalized float in its format. 
		If trapping is enabled, the  floating-point-underflow condition is signalled. Otherwise, the operation results in a denormalized float or zero. 

		:overflow 
		This exception occurs when the result of an operation is too large to be represented as a float in its format. 
		If trapping is enabled, the  floating-point-overflow exception is signalled. Otherwise, the operation results in the appropriate infinity. 

		:inexact 
		This exception occurs when the result of a floating point operation is not exact, i.e. the result was rounded. 
		If trapping is enabled, the extensions:floating-point-inexact condition is signalled. Otherwise, the rounded result is returned. 

		:invalid 
		This exception occurs when the result of an operation is ill-defined, such as (/ 0.0 0.0). 
		If trapping is enabled, the extensions:floating-point-invalid condition is signalled. Otherwise, a quiet NaN is returned. 

		:divide-by-zero 
		This exception occurs when a float is divided by zero. If trapping is enabled, the  divide-by-zero condition is signalled. 
		Otherwise, the appropriate infinity is returned. 
		*/

		unsigned int  myStatus87=0;
		unsigned int  myStatusSSE=0;
		/*
		For _status87 and _statusfp, the bits in the value returned indicate the floating-point status. 
		See the FLOAT.H include file for a complete definition of the bits returned by _status87. 
		Many math library functions modify the 8087/80287 status word, with unpredictable results. 
		Return values from _clear87 and _status87 are more reliable if fewer floating-point operations are performed 
		between known states of the floating-point status word. _statusfp2 has no return value.
		*/
		myStatus87 =_statusfp();

		/*_statusfp2 is recommended for chips (such as the Pentium IV and later) that have both an x87 and an SSE2 floating point processor. 
		For _statusfp2, the addresses are filled in with the floating-point status word for both the x87 or the SSE2 floating-point processor. 
		When using a chip that supports x87 and SSE2 floating point processors, 
		EM_AMBIGUOUS is set to 1 if _statusfp or _controlfp is used and the action was ambiguous because it could refer to the x87 or the SSE2 
		floating-point status word.
		*/

		/*_clearfp is a platform-independent, portable version of the _clear87 routine. 
		It is identical to _clear87 on Intel (x86) platforms and is also supported by the MIPS and ALPHA platforms. 
		To ensure that your floating-point code is portable to MIPS or ALPHA, use _clearfp. If you are only targeting x86 platforms, 
		you can use either _clear87 or _clearfp.*/
		_clearfp();
		myStatus87 =_statusfp(); 
		if (myStatus87 & _SW_INEXACT ) 
			cout<<"We have a floating point problem Houston "<<myStatus87<<endl<<endl;
		else
			cout<<"We have no floating point problems Houston SSE and x87 status is "<<myStatus87<<endl<<endl;

		float z = mariIn;
		float root =kl_arbitrary_precision_root( z,32 ); 
		std::cout.precision(std::numeric_limits<float>::digits);

		cout<<endl<<"Your root is "<<root<<endl<<endl;
		cout<<endl<<"Nice huh?  "<<endl<<endl;

		myStatus87= _statusfp(); 
		if (myStatus87 & _SW_INEXACT ) 
		{
			cout<<"We have checked the floating point status word and the resut is that rounding has occured."<<endl<<endl;
			cout<<"x87 status is "<<myStatus87<<" _SW_INEXACT"<<endl<<endl;
		}

		cout<<endl<<"Now let's check our answer. "<<endl<<"r*r*r ="<<root<<" * "<<root<<" * "<<root<<" = " <<root*root*root<<endl<<endl;

		cout<<"Looks good, but unless you were incredily lucky you got an inexact result.  Homework; under what conditions does the algorithm result in no rounding?"<<endl<<endl;

		cout<<"Let's try a double - that's 64 bit precision. "<<endl<<endl;

		_clearfp();
		double zD= mariIn;
		double rootD = kl_arbitrary_precision_root( zD,64);
		std::cout.precision(std::numeric_limits<double>::digits);

		cout<<endl<<"Your 64 bit root is "<<rootD<<endl;

		myStatus87=_statusfp(); 
		if (myStatus87 & _SW_INEXACT ) 
		{
			cout<<"We have checked the floating point status word and the resut is that rounding has occured."<<endl<<endl;
			cout<<"x87 status is "<<myStatus87<<" _SW_INEXACT"<<endl;
		}		
		cout<<endl<<"Now let's check our answer. "<<endl<<"r*r*r ="<<rootD<<" * "<<rootD<<" * "<<rootD<<" = " <<rootD*rootD*rootD<<endl<<endl;

		cout<<endl<<"Is that better my boo?  "<<endl<<endl;;

		cout<<"Let's try a long double - that's 128 bit precision. "<<endl<<endl;

		_clearfp();
		long double zLD= mariIn;
		long double rootLD = kl_arbitrary_precision_root( zLD,128);
		std::cout.precision(std::numeric_limits<long double>::digits);

		cout<<endl<<"Your 80 bit root is "<<rootLD<<endl<<endl;

		myStatus87=_statusfp(); 
		if (myStatus87 & _SW_INEXACT ) 
		{
			cout<<"We have checked the floating point status word and the resut is that rounding has occured."<<endl<<endl;
			cout<<"x87 status is "<<myStatus87<<" _SW_INEXACT"<<endl<<endl;
		}		
		cout<<"Same Answer?  Yeah, we're not using the Intel Compiler. That's a homework assignment  Read the comments to find out more"<<endl<<endl;
		/*
		On the x86 architecture, most compilers implement long double as the 80-bit extended precision type supported by that hardware 
		(sometimes stored as 12 or 16 bytes to maintain data structure alignment). An exception is Microsoft Visual C++ for x86, 
		which makes long double a synonym for double. The Intel C++ compiler on Microsoft Windows supports extended precision, but 
		requires the /Qlong-double switch to access the hardware's extended precision format.
		Although the x86 architecture, and specifically the x87 floating-point instructions on x86, supports 80-bit 
		extended-precision operations, it is possible to configure the processor to automatically round operations to 
		double (or even single) precision. With gcc on several BSD operating systems (FreeBSD, NetBSD, and OpenBSD), 
		double-precision mode is the default, and long double operations are effectively reduced to double precision. 
		However, it is possible to override this within an individual program via the FLDCW "floating-point load control-word" 
		instruction.[10] Microsoft Windows with Visual C++ also sets the processor in double-precision mode by default, 
		but this can again be overridden within an individual program (e.g. by the _controlfp_s function in Visual C++]). 
		The Intel C++ Compiler for x86, on the other hand, enables extended-precision mode by default.		
		*/



		/*
		Use Intel Intrinsic to calculate cube root 

		//double rootIntelCrbtD=  cbrt(mariIn);
		//long double rootIntelCrbtLD =cbrtl(mariIn);
		//float rootIntelCrbtf =cbrtf(mariIn);


		//cout<<endl<<"But first I'm going to give you the value calculated via Intel Intrisics"<<endl;
		//cout<<"\tfloat val = "<<rootIntelCrbtf<<endl<<endl;
		//cout<<"\tdoube val = "<<rootIntelCrbtD<<endl<<endl;
		//cout<<"\tlong doube val = "<<rootIntelCrbtLD<<endl<<endl;
		*/
		cout<<"Now we are going to use arbitrary precision to find your root."<<endl<<endl;
		cout<<"How many digits would you like? Enter a good number I tried it up to 2048 significands "<<endl<<endl;
		unsigned int precision;
		cin>>precision;
		cout<<"OK, we're going to use "<<precision<<" significands to calculate the root of "<<mariIn<<endl<<endl;

		/*
		NTL by Victor Shoup has fixed and arbitrary high precision fixed and floating-point types. 
		However none of these are licenced for commercial use. 
		#include <NTL/quad_float.h> // quad precision 106-bit, about 32 decimal digits.
		using NTL::to_quad_float; // Less precise than arbitrary precision NTL::RR.
		NTL class quad_float, which gives a form of quadruple precision, 106-bit significand 
		(but without an extended exponent range.) With an IEC559/IEEE 754 compatible processor, 
		for example Intel X86 family, with 64-bit double, and 53-bit significand, using the significands 
		of two 64-bit doubles, if std::numeric_limits<double>::digits10 is 16, then we get about twice the 
		precision, so std::numeric_limits<quad_float>::digits10() should be 32. (the default std::numeric_limits<RR>::digits10() 
		should be about 40). (which seems to agree with experiments). We output constants (including some noisy bits, 
		an approximation to std::numeric_limits<RR>::max_digits10()) by adding 2 extra decimal digits, so using 
		quad_float::SetOutputPrecision(32 + 2); 

		NTL class RR 
		Arbitrary precision floating point with NTL class RR, default is 150 bit (about 50 decimal digits). 
		*/



		LARGE_INTEGER* freq;
		_LARGE_INTEGER* prefCountStart;
		_LARGE_INTEGER* prefCountEnd;
		freq=new _LARGE_INTEGER;
		prefCountStart=new _LARGE_INTEGER;
		prefCountEnd=new _LARGE_INTEGER;
		QueryPerformanceFrequency(freq);
		QueryPerformanceCounter(prefCountStart);
		//for(int precisionIter = 1; precisionIter<precision;precisionIter++)
		{
			int precisionIter = precision;
			RR zRBP;
			zRBP.SetPrecision(precisionIter);
			zRBP.SetOutputPrecision(precisionIter);

			RR rootRBP;
			rootRBP.SetPrecision(precisionIter);
			rootRBP.SetOutputPrecision(precisionIter);

			zRBP = mariIn;

			//clock_t processMilisec = clock();
			QueryPerformanceCounter(prefCountStart);
			rootRBP = kl_arbitrary_precision_root( zRBP,precisionIter);
			QueryPerformanceCounter(prefCountEnd);
			//rootRBP = kl_arbitrary_precision_root( zRBP,precision);
			//processMilisec = clock() - processMilisec;
			//cout<<precisionIter<<" : "<<processMilisec<< " ";
			cout<<"Runtime (millisec) for precision "<<precisionIter<<" = "<<double(prefCountEnd->QuadPart-prefCountStart->QuadPart)/double(freq->QuadPart)<<endl;   
			cout<<"Your root is "<<rootRBP<<endl;
			cout<<"And  r*r*r ="<<endl<<endl<<rootRBP<<" * "<<endl<<endl<<rootRBP<<" * "<<endl<<endl<<rootRBP<<" = "<<endl<<endl<<rootRBP*rootRBP*rootRBP<<endl<<endl;
		}

		cout<<"Enter anything other than 0 if you want to go again : "<<endl;

		cin>>mariIn;

	}while(mariIn>0);

	return 0;
}

#include <NTL/ZZ.h>

using namespace std;
using namespace NTL;

long witness(const ZZ& n, const ZZ& x)
{
   ZZ m, y, z;
   long j, k;

   if (x == 0) return 0;

   // compute m, k such that n-1 = 2^k * m, m odd:

   k = 1;
   m = n/2;
   while (m % 2 == 0) {
      k++;
      m /= 2;
   }

   z = PowerMod(x, m, n); // z = x^m % n
   if (z == 1) return 0;

   j = 0;
   do {
      y = z;
      z = (y*y) % n; 
      j++;
   } while (j < k && z != 1);

   return z != 1 || y != n-1;
}

long PrimeTest(const ZZ& n, long t)
{
   if (n <= 1) return 0;

   // first, perform trial division by primes up to 2000

   PrimeSeq s;  // a class for quickly generating primes in sequence
   long p;

   p = s.next();  // first prime is always 2
   while (p && p < 2000) {
      if ((n % p) == 0) return (n == p);
      p = s.next();  
   }

   // second, perform t Miller-Rabin tests

   ZZ x;
   long i;

   for (i = 0; i < t; i++) {
      x = RandomBnd(n); // random number between 0 and n-1

      if (witness(n, x)) 
         return 0;
   }

   return 1;
}

int testMaxCampbellPrime()
{
   const int bitsToUse =63;
   //int powers[bitsToUse] = {0,1,0,0,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,0,1,1,0,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0};
   int powers[bitsToUse] ={1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,1,1,0,0,0,0,1,0,1,1,0,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,0};   
   ZZ n;//ZZ() initial value is 0.
   for(int i=0;i<95;i++)
   {
	   if (powers[i]==1)
	   {
		   long expi = i;
		   long two = 2;
		   ZZ n2 =power_ZZ(2,i); 

		   n=n+n2;
		   cout <<n<<endl;
	   }
   }
   if (PrimeTest(n, 10))
      cout << n << " is probably prime\n";
   else
      cout << n << " is composite\n";
}

#include <limits.h>