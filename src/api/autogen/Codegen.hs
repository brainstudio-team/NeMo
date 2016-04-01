{- General code-generation utilities -}

module Codegen (
    Code,
    nl,
    indent,
    op,
    assign,
    insertFileContents,
    bracket,
    cbracket,
    list,
    arglist)
 where

import Text.PrettyPrint
import Data.List (intercalate)
import System.IO
import Text.Printf

import API

type Code = String
type Indent = Int


nl :: Code -> Code
nl str = str ++ "\n"

indent :: Int -> Code -> Code
indent n str = replicate (n*4) ' ' ++ str

op :: Code -> Code -> Code -> Code
op opname lhs rhs = printf "%s %s %s" lhs opname rhs

assign = op "="

wrap :: Char -> Char -> Code -> Code
wrap open close code = open : code ++ [close]

bracket :: Code -> Code
bracket = wrap '(' ')'
-- bracket code = "(" ++ code ++ ")"

cbracket :: Code -> Code
cbracket = wrap '{' '}'
-- cbracket code = "{" ++ code ++ "}"



insertFileContents :: Handle -> String -> IO ()
insertFileContents h filename = hPutStr h =<< hGetContents =<< openFile filename ReadMode


{- | Return a pretty-printed comma-separated list. If the list is long it's
 - printed with on argument per-line with the given indentation (in tabs) -}
list :: Indent -> (Code -> Code) -> [Code] -> Code
list ntabs wrapper args =
    if baseLen <= 60
        then wrapper baseList
        else wrapper $ "\n" ++ (intercalate ",\n" $ map prefixTabs args)
    where
        baseList = intercalate ", " args
        baseLen = length baseList
        prefixTabs str = replicate ntabs '\t' ++ str


{- | Return a pretty-printed argument list. If the list is long it's printed
 - with on argument per-line with the given indentation (in tabs) -}
arglist :: Indent -> [Code] -> Code
arglist ntabs args = list ntabs bracket args
{-
    if baseLen <= 60
        then bracket $ baseList
        else bracket $ "\n" ++ (intercalate ",\n" $ map prefixTabs args)
    where
        baseList = intercalate ", " args
        baseLen = length baseList
        prefixTabs str = replicate ntabs '\t' ++ str
        -}



