namespace FSom

open System.Collections.Generic
open System.Diagnostics

[<AutoOpen>]
module point =
    [<DebuggerDisplay("({xy})")>]
    type Point =
    |Point of X : int * Y : int

        member this.xy = 
            match this with
            | Point(X, Y) -> X, Y
        
        member this.x =
            match this with
            |Point(X = x) -> x

        member this.y =
            match this with
            |Point(Y = y) -> y

        //neighbors of a point on a rectangular grid
        member this.getNeighbors width height =
            let row, col = this.xy
            let neighbors = List<Point>()
            for i = -1 to 1 do
                for j = -1 to 1 do
                    if (i <> 0 || j <> 0) && i + row >= 0 && j + col >= 0 && i + row < height && j + col < width then
                        neighbors.Add(Point(row + i, col + j))
            neighbors

        member this.addNeighbors width height =
            let neighbors = this.getNeighbors width height
            neighbors.Add(this)
            neighbors

        override this.ToString () =
            this.x.ToString() + "," + this.y.ToString()

       static member Copy (point : Point) = Point(point.xy)


