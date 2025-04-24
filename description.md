Introduction:
The first step in automated trading is to process a stream of market data that is disseminated by the exchange. This test will simulate processing such feed
(with the exception that for this exercise the raw data is read from file vs from a socket). Please note that we read from the file for you and you work with the data all loaded in memory.
The information stream contains two types of data:
a. Information about quotes of different symbols in the market, for example GOOG is now quoted at $ 614.33 bid (i.e. to buy) and $614.34 ask (i.e. to sell)
b. Information about trades in the market, for example 100 shares of GOOG were just traded at $614.34.
In this exercise you are asked to read such raw information as it has been captured in a file, and print out the trades that were captured during that time.
Specifications:
• The input data is provided in a memory array (already read from the binary file).
• The data contains packets of market data starting with the first byte of the array. All Integers are unsigned and in big-endian.
Each packet has the following header:
Name    Offset  Num Bytes   Type    Description     
Packet Length   0   2   Integer Length of the entire packet (Including this data member).
Num Market
Updates
2
2
Integer
Number of market updates in this packet.