module Common where

import Text.PrettyPrint
import Data.List
import Data.Char


-- vcat with additional space
vsep :: [Doc] -> Doc
vsep xs = vcat $ intersperse (text "") $ xs


mergeLinesWith :: String -> [Doc] -> Doc
mergeLinesWith str = hcat . punctuate (text str)

mergeLines :: [Doc] -> Doc
mergeLines = mergeLinesWith "\\n"





camelToUnderscore :: String -> String

camelToUnderscore [] = ""
camelToUnderscore (x:xs) = toLower x : go xs
    where
        go [] = []
        go (x:xs)
            | isUpper x = '_' : toLower x : go xs
            | otherwise = x : go xs


-- different function naming schemes
underscoredName :: [String] -> Doc
underscoredName xs = text $ map toLower $ intercalate "_" xs


underscoredUpper :: [String] -> Doc
underscoredUpper xs = text $ map toUpper $ intercalate "_" xs


camelCasedName :: [String] -> String
camelCasedName [] = ""
camelCasedName (x:xs) = x ++ (concatMap (firstCap) xs)
    where
        firstCap :: String -> String
        firstCap [] = []
        firstCap (x:xs) = toUpper x : xs


listPunctuate :: [Doc] -> Doc
listPunctuate = hcat . punctuate (text ", ")


arglistWith :: (Doc -> Doc) -> [Doc] -> Doc
arglistWith bracket = bracket . listPunctuate


arglist = arglistWith parens


{- Long lines can be wrapped in different ways -}
data LineWrap
    = None        -- ^ never wrap
    | Compact Doc -- ^ wrap as little as possible, using the specified string to end each line
    | Full Doc    -- ^ one element per line, using the specified string to end each line


-- TODO: Add function definition as well
functionGeneric
    :: Doc          -- ^ symbol between 'return' and 'name'
    -> LineWrap     -- ^ wrapping mode
    -> Doc          -- ^ function name
    -> Maybe Doc    -- ^ output assignment
    -> Int          -- ^ allowed text width before wrapping
    -> [Doc]        -- ^ arguments
    -> Doc
functionGeneric assignOp wrap name output width inputs = assign <> (call wrap')
    where
        assign = maybe empty (<> assignOp) output

        wrap' = if length args_oneline <= width then None else wrap

        call None           = (<>) name $ parens $ text args_oneline
        -- break only after comma, but as late as possible
        call (Compact cont) = fcat $ [name <> lparen <> cont] ++ (args_multiline cont) ++ [rparen]
        call (Full cont)    = vcat $ [name <> lparen <> cont] ++ (args_multiline cont) ++ [rparen]

        -- TODO: separate out multiline functionality
        args_oneline = render $ hcat $ punctuate (text ", ") inputs
        args_multiline cont = map (nest 8 . (<> cont)) $ punctuate (text ", ") inputs


functionCall
    :: LineWrap
    -> Doc          -- ^ function name
    -> Maybe Doc    -- ^ output assignment
    -> Int          -- ^ allowed text width before wrapping
    -> [Doc]        -- ^ arguments
    -> Doc
functionCall = functionGeneric (text " = ")


functionDefinition
    :: LineWrap
    -> Doc          -- ^ function name
    -> Maybe Doc    -- ^ output assignment
    -> Int          -- ^ allowed text width before wrapping
    -> [Doc]        -- ^ arguments
    -> Doc
functionDefinition = functionGeneric space

angleBrackets :: Doc -> Doc
angleBrackets p = char '<' <> p <> char '>'


{- Sometimes 'empty' does the wrong thing -}
emptyLine :: Doc
emptyLine = text ""
