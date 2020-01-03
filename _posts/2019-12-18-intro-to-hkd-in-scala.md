---
layout: post
title:  "[DRAFT] A quick and gentle introduction to higher kinded data in Scala"
date:   2019-12-18 13:00:00 +0100
categories: scala hkd
---
{% include code_blocks_init.html %}

In Scala, we have higher kinded types. Using these, we can abstract over types 
that themselves take types, like `List`. You've probably seen higher kinded 
types used in typeclasses like `Functor` and friends, or in tagless final. 
Their usage often looks like this.
```scala
trait Functor[F[_]] {
  def map[A, B](fa: F[A])(f: A => B): F[B]
}

trait UserService[F[_]] {
  def getUser(id: UserId): F[User]
  
  def updateUser(user: User): F[Unit]
}
```

Today, however, I want to talk about another usage of higher kinded types 
called higher kinded data (HKD). Note that I do assume you're at least a bit familiar 
with higher kinded types, and the most common typeclasses that use them like functor, 
applicative, monad, and so on. I'm also assuming that you know what kind 
projector is and why it's useful. This post would likely be twice as long if I
would have to go over these things too, and there are better posts out there 
about them than what I can provide here.

## A mountain of boilerplate

Say that you're working on a PATCH endpoint. You take a partial json body, and 
change some stuff in the database for the provided values. Let's use an 
additional `Option`, to indicate if a JSON property was `undefined`.
```scala
//For this post, I'll use Circe and doobie for Json and SQL
def patchProject(projectId: String, json: Json): IO[Result] = {
  val root = json.hcursor
  val settings = root.downField("settings")
  
  val partialProjectResult = (
    withUndefined[String]("name", root),
    withUndefined[String]("description", root),
    withUndefined[List[String]]("keywords", settings),
    withUndefined[Option[String]]("issues", settings),
    withUndefined[Option[String]]("sources", settings),
    withUndefined[Option[String]]("support_channel", settings),
  ).mapN(PartialProject.apply)
  
  
  //Imagine what handleDecodeError looks like yourself
  partialProjectResult.fold(handleDecodeError) { partialProject =>
    val sets = Fragments.setOpt(
      partialProject.name.map(name => fr"name = $name"),
      partialProject.description.map(description => fr"description = $description"),
      partialProject.keywords.map(keywords => fr"keywords= $keywords"),
      partialProject.issues.map(issues => fr"issues = $issues"),
      partialProject.sources.map(sources => fr"sources = $sources"),
      partialProject.supportChannel.map(supportChannels => fr"support_channel = $supportChannel"),
    )
  
    val query = sql"UPDATE projects  " ++ sets ++ fr"WHERE project_id = $projectId"
  
    val hasAnyUpdates = partialProject.name.isDefined || 
      partialProject.description.isDefined || partialProject.keywords.isDefined || 
      partialProject.issues.isDefined || partialProject.sources.isDefined || 
      partialProject.supportChannel.isDefined
  
    if (hasAnyUpdates) dbService.run(query.update.run).map(_ => NoContent)
    else IO.pure(BadRequest("No updates specified"))
  }
}

def withUndefined[A: Decoder](
    field: String, cursor: ACursor
): Decoder.AccumulatingResult[Option[A]] = {
  val result = if(cursor.succeeded) Some(cursor.get[A](field)) else None
  result.sequence.toValidatedNel
}

@SnakeCaseDecoder case class Project(
  name: String,
  description: String,
  keywords: List[String],
  issues: Option[String],
  sources: Option[String],
  supportChannel: Option[String]
)

case class PartialProject(
  name: Option[String],
  description: Option[String],
  keywords: Option[List[String]],
  issues: Option[Option[String]],
  sources: Option[Option[String]],
  supportChannel: Option[Option[String]]
)
```

That's a lot of boilerplate just for a PATCH. Worse, there are more PATCH 
endpoints like this one that needs to be defined. Is there a better way to do this?
We could maybe get rid of some of the boilerplate if we created a special 
typeclass to decode partial data into case classes, and then use that on the 
`PartialProject` class. That would only get rid of some of the boilerplate though.
Wouldn't it be great if we could just create a method `handlePatch` that does 
all the work for us? Could maybe be used something like this `handlePatch[Project]`. 
Let's do a rough draft of what that might look like. We'll replace all cases 
of boilerplate with `...`. Let's also parameterize all the cases where we 
refer to something specific.

```scala
def handlePatch[Thing](
    table: String, 
    identifierColumn: String, 
    identifier: String, 
    json: Json
): IO[Result] = {
  val partialThingResult = (
    ...
  ).mapN(PartialThing.apply)
  
  
  partialThingResult.fold(handleDecodeError) { partialThing =>
    val sets = Fragments.setOpt(
        ...
    )
  
    val query = fr"UPDATE" ++ Fragment.const(table) ++ sets ++ fr"WHERE" ++ 
      Fragment.const(identifierColumn) ++ fr"= $identifier"
  
    val hasAnyUpdates = ...
  
    if (hasAnyUpdates) dbService.run(query.update.run).map(_ => NoContent)
    else IO.pure(BadRequest("No updates specified"))
  }
}
```

Okay, that's not as bad. If we just sprinkle in some shapeless records and 
loop over it a few times, we should have the method we want, right? Yes, that 
would work, but it also lands us right into logic programming land. Shapeless is 
nice for derivation of typeclasses and such, where you can just trust the type 
signature. It's a bit worse in actual application code, where you want to read 
what is actually happening, and can't just look at the types. Is there another 
way to solve this problem that does not include using shapeless at all, and 
shows more of our intent? Yes, there is.

## Introducing ProjectF
Look back at the boilerplate mountain. We wrote two case classes, with the 
same fields, just that one had everything wrapped in `Option`. Let's instead 
define a single case class with a higher kinded type parameter, which 
indicates the wrapping type.
```scala
case class ProjectF[F[_]](
  name: F[String],
  description: F[String],
  // Note that I don't wrap this in F, as it's content will be wrapped in F 
  // instead. I might talk about when you also want to wrap this in F at a 
  // later point
  settings: ProjectSettingsF[F]
)

case class ProjectSettingsF[F[_]](
  keywords: F[List[String]],
  issues: F[Option[String]],
  sources: F[Option[String]],
  supportChannel: F[Option[String]]
)
```

This is the one of the cornerstones of higher kinded data (HKD), the data itself. 
Note that I split up the class into two smaller classes to mirror the JSON
structure. You can go either way here, and it shouldn't matter for the patch 
method, as long as the parameters you'll pass in are the same. I'll talk a bit more about this later.

We can now get back `PartialProject` like so.
```scala
type PartialProject = ProjectF[Option]
```

Can we also use this case class for the non-partial case class? Yes, using the 
`Id` type. The `Id` type just spits back out what we throw at it. Think of it 
like a type level `Predef.identity`.
```scala
type Id[A] = A
type Project = ProjectF[Id]
```

### Const is a really useful type

There is one more important type we need, `Const`. (When the 
Scala and Dotty representation of a concept differs substantially, I'll 
include both.)
{% capture scala-const %}
// Here we're defining a partially applied type. We use it like so 
// Const[String]#λ[Int], or if we're in a place expecting a higher kinded 
// type, just Const[String]#λ
type Const[A] = {
  type λ[B] = A
}
{% endcapture %}

{% capture dotty-const %}
type Const[A] = [B] =>> A
{% endcapture %}

{% include code_blocks_code.html scala=scala-const dotty=dotty-const id="const-type" %}

It is, in some ways, opposite to how `Id` works. While `Id` spits back what we 
threw at it, `Const` ignores that and instead spits back what it was initially 
applied with. Here's an example.
{% capture scala-const-usage %}
type Name[A] = Const[String]#λ[A]

type Foo = Name[Int] // Type of Foo is String
type Bar = Name[String] // Type of Bar is String
type Baz = Name[Option[List[Double]]] // Type of Baz is String
type Bin = Name[Nothing] // Type of Bin is String
{% endcapture %}

{% capture dotty-const-usage %}
// In dotty we don't really need to apply Const with A here.
// I do it here for less confusion.
type Name[A] = Const[String][A]

type Foo = Name[Int] // Type of Foo is String
type Bar = Name[String] // Type of Bar is String
type Baz = Name[Option[List[Double]]] // Type of Baz is String
type Bin = Name[Nothing] // Type of Bin is String
{% endcapture %}

{% include code_blocks_code.html scala=scala-const-usage dotty=dotty-const-usage id="const-type-usage" %}
What's so important about `Const`? It allows us to put any type into `ProjectF` 
we want, as long as it's the same everywhere. `ProjectF[Const[A]]` therefore 
becomes similar to something like this.

```scala
case class ProjectConst[A](
  name: A,
  description: A,
  // Again we don't say A here, but instead ProjectSettingsConst[A]
  settings: ProjectSettingsConst[A]
)

case class ProjectSettingsConst[A](
  keywords: A,
  issues: A,
  sources: A,
  supportChannel: A
)
```

### Names for fields with HKD

Why is this useful? Because it allows us to essentially "tag" data with which 
field it belongs to. For example, we could have a `ProjectF[Const[String]]` 
instance where each field contains the name of the field. Could we use a 
`ProjectF[Const[String]]`instance where the fields contains their names to 
store the json field names? No, we're still missing something. In such an 
instance, the value of `project.settings.issues` would be `"issues"`, but that 
completely ignores the settings field. The fix for that is simple, instead 
of `Const[String]`, let's use `Const[List[String]]`. Giving such an instance 
for `ProjectF` gives us this. Note that if you choose to keep everything in 
one class instead of splitting them in two classes, you still end up with 
roughly the same structure. (This is our first hint at how nested HKD differs 
from normal data. More on nested and flat HKD much later.)

{% capture scala-projectF-names %}
object ProjectF {
  val names: ProjectF[Const[List[String]]#λ] = ProjectF[Const[List[String]]#λ](
    List("name"),
    List("description"),
    ProjectSettingsF[Const[List[String]]#λ](
      List("settings", "keywords"),
      List("settings", "issues"),
      List("settings", "sources"),
      List("settings", "supportChannel")
    )
  )
}
{% endcapture %}

{% capture dotty-projectF-names %}
object ProjectF with
  val names: ProjectF[Const[List[String]]] = ProjectF(
    List("name"),
    List("description"),
    ProjectSettingsF(
      List("settings", "keywords"),
      List("settings", "issues"),
      List("settings", "sources"),
      List("settings", "supportChannel")
    )
  )
{% endcapture %}

{% include code_blocks_code.html scala=scala-projectF-names dotty=dotty-projectF-names id="projectF-names" %}
This is one of the things it can be nice to have a macro generate, but for now
, we'll write it out manually. Anyway, that's pretty nice, just one problem. 
In many of our cases, we're using `snake_case`. We could just redefine 
`ProjectF`, but what if we instead made a function that transforms the strings 
in the structure?

{% capture scala-projectF-names-transform %}
object ProjectF {
  val names = ...
  
  def transformNames(oldNames: ProjectF[Const[List[String]]#λ])(
      f: String => String
  ): ProjectF[Const[List[String]]#λ] = ProjectF[Const[List[String]]#λ](
    oldNames.name.map(f),
    oldNames.description.map(f),
    ProjectSettingsF[Const[List[String]]#λ](
      oldNames.settings.keywords.map(f),
      oldNames.settings.issues.map(f),
      oldNames.settings.sources.map(f),
      oldNames.settings.supportChannel.map(f)
    )
  )
  
  // Imagine yourself where snakeCaseRename comes from
  val snakeCaseNames: ProjectF[Const[List[String]]#λ] = transformNames(names)(snakeCaseRename)
}
{% endcapture %}

{% capture dotty-projectF-names-transform %}
object ProjectF with
  val names = ...
  
  def transformNames(oldNames: ProjectF[Const[List[String]]])(
      f: String => String
  ): ProjectF[Const[String]#λ] = ProjectF(
    oldNames.name.map(f),
    oldNames.description.map(f),
    ProjectSettingsF(
      oldNames.settings.keywords.map(f),
      oldNames.settings.issues.map(f),
      oldNames.settings.sources.map(f),
      oldNames.settings.supportChannel.map(f)
    )
  )
  
  // Imagine yourself where snakeCaseRename comes from
  val snakeCaseNames: ProjectF[Const[List[String]]] = transformNames(names)(snakeCaseRename)
{% endcapture %}

{% include code_blocks_code.html scala=scala-projectF-names-transform dotty=dotty-projectF-names-transform id="projectF-names-transform" %}

## Let's implement some typeclasses

Wait... We just applied a function over the entire structure. Can we do this 
with any type? Isn't that what a functor is? Yes, and `ProjectF[Const]` has 
one. Let's define it.

{% capture scala-projectF-const-functor %}
object ProjectF {
  ... // All the stuff we defined before
   
  implicit val projectConstFunctor: Functor[λ[A => ProjectF[Const[A]#λ]]] = 
    new Functor[λ[A => ProjectF[Const[A]#λ]]] {
      override def map[A, B](fa: ProjectF[Const[A]#λ])(f: A => B): ProjectF[Const[B]#λ] = 
        ProjectF[Const[B]#λ](
          f(fa.name),
          f(fa.description),
          ProjectSettingsF[Const[B]#λ](
            f(fa.keywords),
            f(fa.issues),
            f(fa.sources),
            f(fa.supportChannel)
          )
        )
    }
}
{% endcapture %}

{% capture dotty-projectF-const-functor %}
object ProjectF with
  ... // All the stuff we defined before
   
  given Functor[[A] =>> ProjectF[Const[A]]]:
    override def [A, B](fa: ProjectF[Const[A]]) map(f: A => B): ProjectF[Const[B]] = ProjectF(
      f(fa.name),
      f(fa.description),
      ProjectSettingsF(
        f(fa.keywords),
        f(fa.issues),
        f(fa.sources),
        f(fa.supportChannel)
      )
    )
{% endcapture %}

{% include code_blocks_code.html scala=scala-projectF-const-functor dotty=dotty-projectF-const-functor id="projectF-const-functor" %}

Okay, so we got some nice abstraction for `Const`. Can we generalize it 
further and what would that look like? Currently, we have a `map` function that 
takes in a `ProjectF[Const[A]]`, and returns a `ProjectF[Const[B]]`. 
What if we could instead define a function that takes a `ProjectF[A]`, and 
returns a `ProjectF[B]`, where `A` and `B` are higher kinded types? That 
sounds like a functor on `ProjectF`. Before we define this, we need yet 
another type. `A => B` just won't be enough anymore.

### Natural transformations
What we need is to somehow be able to pass something like this in as a value.
```scala
def headOption[A](xs: List[A]): Option[A] = xs.headOption
```

We can pass `List[Int] => Option[Int]` and `List[String] => Option[String]` 
as values, but `List => Option` isn't valid. Luckily there is a way to encode 
what we want. We can define a new type `FunctionK`, and alias it to `~>:`. 
I throw on a `:` here as I prefer my arrows to associate in the right direction.

{% capture scala-functionK %}
trait FunctionK[A[_], B[_]] {
  def apply[Z](a: A[Z]): B[Z]
}
object FunctionK {
  
  def identity[F[_]]: F ~>: F = λ[F ~>: F](Predef.identity(_))

  def const[F[_], A](a: A): F ~>: Const[A]#λ = new FunctionK[F, Const[A]#λ] {
    override def apply[Z](fz: F[Z]): A = a
  }
}

// Stick this in some package object somewhere
type ~>:[A[_], B[_]] = FunctionK[A, B]
{% endcapture %}

{% capture dotty-functionK %}
// Luckily Dotty already has an encoding for these, 
// so we'll just add a few type aliases 
type FunctionK[A[_], B[_]] = [Z] => A[Z] => B[Z]
type ~>:[A[_], B[_]] = FunctionK[A, B]

object FunctionK with
  def identity[F[_]]: F ~>: F = [Z] => (fz: F[Z]) => fz

  def const[F[_], A](a: A): F ~>: Const[A] = [Z] => (fz: F[Z]) => a
{% endcapture %}

{% include code_blocks_code.html scala=scala-functionK dotty=dotty-functionK id="functionK" %}

We can create and use them like this.
{% capture scala-functionK-usage %}
// Normal usage looks like so. We need to create a new instance of the class in 
// the same way you had for functions in Java before Java 8.
val headOption1: List ~>: Option = new (List ~>: Option) {
  override def apply[Z](fa: List[Z]): Option[Z] = fa.headOption
}

// We can however also use Kind projector in simple cases, but then we loose the 
// ability to refer to the type.
val headOption2: List ~>: Option = λ[List ~>: Option](_.headOption)

val optHead1: Option[Int] = headOption1(Nil)
val optHead2: Option[Int] = headOption2(Nil)
{% endcapture %}

{% capture dotty-functionK-usage %}
// No underscore syntax here. You must define both the type, and the 
// parameter with the type applied. 
val headOption: List ~>: Option = [Z] => (a: List[Z]) => a.headOption

val optHead: Option[Int] = headOption(Nil)
{% endcapture %}

{% include code_blocks_code.html scala=scala-functionK-usage dotty=dotty-functionK-usage id="functionK-usage" %}

### FunctorK
We now have almost all the pieces we need. We just need a new functor 
typeclass which can handle our new types.

{% capture scala-functorK %}
trait FunctorK[F[_[_]]] {
  def mapK[A[_], B[_]](fa: F[A])(f: A ~>: B): F[B]
  
  def liftK[A[_], B[_]](f: A ~>: B): F[A] => F[B] = mapK(_)(f)
}
{% endcapture %}

{% capture dotty-functorK %}
trait FunctorK[F[_[_]]] with
  def [A[_], B[_]](fa: F[A]) mapK(f: A ~>: B): F[B]
  
  def liftK[A[_], B[_]](f: A ~>: B): F[A] => F[B] = _.mapK(f)
{% endcapture %}

{% include code_blocks_code.html scala=scala-functorK dotty=dotty-functorK id="functorK" %}

Creating higher kinded typeclasses is normally pretty easy and generally just 
involves raising all the kinds of the types by one. `F[_]` becomes `F[_[_]]`, 
`A` becomes `A[_]`, and so on.

(NOTE: While it won't matter too much for this post, I should not that the 
above typeclass is not the one I use myself most of the time. Most notably, 
it's not an endofunctor. More on that later at some point. Why isn't in an 
endofunctor? In an endofunctor, the arrow in `lift` is the same type of arrow 
as the one returned, while here we take an `~>:` arrow and return an `=>` arrow.)

### ApplyK
Great, we have our `FunctorK` typeclass. Let's get a few more. Next up is 
`ApplyK`, but before we create that one, we need to encode arity and tuples.

{% capture scala-tupleK %}
type Tuple2K[F[_], G[_]] = { 
  type λ[A] = (F[A], G[A]) 
}

// While it would be great to define something like Function2K, kind projector 
// only supports functions of arity 1
{% endcapture %}

{% capture dotty-tupleK %}
type Tuple2K[A[_], B[_]] = [Z] =>> (A[Z], B[Z])

// Arity 0 and arity 2 is enough for this case
type ValueK[A[_]] = [Z] => () => A[Z]
type Function2K[A[_], B[_], C[_]] = [Z] => (A[Z], B[Z]) => C[Z]
{% endcapture %}

{% include code_blocks_code.html scala=scala-tupleK dotty=dotty-tupleK id="tupleK" %}

This type lets us have the same type, but with different wrappers. For example, using `ProjectF[Tuple2K[List, Option]]` gives us something like this.
```scala
case class TupledProject(
  name: (List[String], Option[String]),
  description: (List[String], Option[String]),
  settings: TupledProjectSettings
)

case class TupledProjectSettings(
  keywords: (List[List[String]], Option[List[String]]),
  issues: (List[Option[String]], Option[Option[String]]),
  sources: (List[Option[String]], Option[Option[String]]),
  supportChannel: (List[Option[String]], Option[Option[String]])
)
```
With that out of the way, let's define `ApplyK`. This is probably the most 
useful typeclass for HKD I think.

{% capture scala-applyK %}
trait ApplyK[F[_[_]]] extends FunctorK[F] {
  def apK[A[_], B[_]](ff: F[λ[D => A[D] => B[D]]])(fa: F[A]): F[B] =
    map2K(ff, fa)(λ[Tuple2K[λ[D => A[D] => B[D]], A]#λ ~>: B](t => t._1(t._2)))

  def tupledK[A[_], B[_]](fa: F[A], fb: F[B]): F[Tuple2K[A, B]#λ] =
    map2K(fa, fb)(FunctionK.identity)

  def map2K[A[_], B[_], Z[_]](fa: F[A], fb: F[B])(f: Tuple2K[A, B]#λ ~>: Z): F[Z]
}
{% endcapture %}

{% capture dotty-applyK %}
trait ApplyK[F[_[_]]] extends FunctorK[F]
  def [A[_], B[_]](ff: F[[D] =>> A[D] => B[D]]) apK(fa: F[A]): F[B] =
    ff.map2K(fa)([Z] => (f: A[Z] => B[Z], az: A[Z]) => f(az))

  def [A[_], B[_], Z[_]](fa: F[A]) map2K(fb: F[B])(f: Function2K[A, B, Z]): F[Z]

  def [A[_], B[_]](fa: F[A]) tupledK(fb: F[B]): F[Tuple2K[A, B]] = 
    fa.map2K(fb)([Z] => (az: A[Z], bz: B[Z]) => (az, bz))
{% endcapture %}

{% include code_blocks_code.html scala=scala-applyK dotty=dotty-applyK id="applyK" %}

### ApplicativeK
We can also define the least useful typeclass, `ApplicativeK`. Why is it so 
useless? Because unlike with the normal applicative, there aren't many cases 
where we want to construct a new instance of our type. In fact, doing so is 
hard because we need to be able to construct `A[Z]`, for all types `Z`. Either 
you can use `Const`, `A[Nothing]` where `A` is covariant, or `A[Any]`, where 
`A` is contravariant. It also gives us nothing more compared to `Applicative[F[Const]]`.

{% capture scala-applicativeK %}
trait ApplicativeK[F[_[_]]] extends ApplyK[F] {

  def pureK[A[_]](a: Const[Unit]#λ ~>: A): F[A]

  def unitK: F[Const[Unit]#λ] = pureK(FunctionK.identity)

  override def mapK[A[_], B[_]](fa: F[A])(f: A ~>: B): F[B] =
    apK(pureK[λ[D => A[D] => B[D]]](λ[Const[Unit]#λ ~>: λ[D => A[D] => B[D]]](_ => f.apply)))(fa)
}
{% endcapture %}

{% capture dotty-applicativeK %}
trait ApplicativeK[F[_[_]]] extends ApplyK[F]
  def [A[_]](a: ValueK[A]) pureK: F[A]

  def unitK: F[Const[Unit]] = ValueK.const(()).pure

  override def [A[_], B[_]](fa: F[A]) mapK(f: A ~>: B): F[B] =
    ([Z] => () => f[Z]).pure[[D] =>> A[D] => B[D]].apK(fa)
{% endcapture %}

{% include code_blocks_code.html scala=scala-applicativeK dotty=dotty-applicativeK id="applicativeK" %}

## First step away from the boilerplate
While we're still missing a few pieces that we're going to need, we can begin 
to look at how we can use these typeclasses to get rid of the boilerplate. 

### Patch decoding
Let's focus on this piece of code first.
```scala
val root = json.hcursor
val settings = root.downField("settings")

val partialProjectResult = (
  withUndefined[String]("name", root),
  withUndefined[String]("description", root),
  withUndefined[List[String]]("keywords", json),
  withUndefined[Option[String]]("issues", json),
  withUndefined[Option[String]]("sources", json),
  withUndefined[Option[String]]("support_channel", json),
).mapN(PartialProject.apply)
```

What do we need here? We need the names which we already have and the decoders. 
Let's make a new instance of `ProjectF` filled with decoders.
```scala
val projectDecoders: ProjectF[Decoder] = ProjectF[Decoder](
  Decoder[String],
  Decoder[String],
  ProjectSettingsF[Decoder](
    Decoder[List[String]],
    Decoder[Option[String]],
    Decoder[Option[String]],
    Decoder[Option[String]],
  )
)
```

We also need the cursor to use. Take all that, blend it together, and we get a 
method to decode an HKD type from a patch payload.

{% capture scala-patch-decode-wrong %}
def patchDecode[F[_[_]]](names: F[Const[List[String]]#λ], decoders: F[Decoder], cursor: ACursor)(
    implicit F: ApplyK[F]
): F[λ[A => Decoder.AccumulatingResult[Option[A]]]] =
  F.map2K(names, decoders)(
    new (Tuple2K[Const[List[String]]#λ, Decoder]#λ ~>: λ[A => Decoder.AccumulatingResult[Option[A]]]) {
      override def apply[Z](t: (List[String], Decoder[Z])): Decoder.AccumulatingResult[Option[Z]] = {
        val names   = t._1
        val decoder = t._2

        val cursorWithNames = names.foldLeft(cursor)(_.downField(_))

        val result = 
          if (cursorWithNames.succeeded) Some(decoder.decode(cursorWithNames)) 
          else None
        result.sequence.toValidatedNel
      }
    }
  )
{% endcapture %}

{% capture dotty-patch-decode-wrong %}
def patchDecode[F[_[_]]](names: F[Const[List[String]]], decoders: F[Decoder], cursor: ACursor)(
    given ApplyK[F]
): F[[A] =>> Decoder.AccumulatingResult[Option[A]]] =
  names.map2K(decoders) { [Z] => (names: List[String], decoders: Decoder[Z]) =>
    val cursorWithNames = names.foldLeft(cursor)(_.downField(_))

    val result = 
      if cursorWithNames.succeeded then Some(decoder.decode(cursorWithNames)) 
      else None
    result.sequence.toValidatedNel
  }
{% endcapture %}

{% include code_blocks_code.html scala=scala-patch-decode-wrong dotty=dotty-patch-decode-wrong id="patch-decode-wrong" %}

What we're doing here is that for each field, we first fold over the names of 
the field, accumulating the result in the cursor. We then check if there is 
something at that field, and if there is, decode it using the decoder.

Wonderful, this is what we want, right? Almost. Just one problem left. Using 
this with `ProjectF`, it gives us a 
`ProjectF[λ[A => Decoder.AccumulatingResult[Option[A]]]]`, but what we want is 
a `Decoder.AccumulatingResult[ProjectF[Option]]`. That sounds like a call to 
`sequence`. Guess we'll need `TraverseK` too. 

### doobie sets
Before we go off and do `TraverseK` too, let's look at some of the other boilerplate.

```scala
val sets = Fragments.setOpt(
  partialProject.name.map(name => fr"name = $name"),
  partialProject.description.map(description => fr"description = $description"),
  partialProject.keywords.map(keywords => fr"keywords= $keywords"),
  partialProject.issues.map(issues => fr"issues = $issues"),
  partialProject.sources.map(sources => fr"sources = $sources"),
  partialProject.supportChannel.map(supportChannels => fr"support_channel = $supportChannel"),
)

val hasAnyUpdates = partialProject.name.isDefined || 
  partialProject.description.isDefined || partialProject.keywords.isDefined || 
  partialProject.issues.isDefined || partialProject.sources.isDefined || 
  partialProject.supportChannel.isDefined
```

`hasAnyUpdates` is probably the easiest one here, and the only one we could 
technically solve (if we erased some types) right now. Say that we somehow 
could convert any `ProjectF[A]` into a `List[A[_]]`. In that case, the problem 
becomes easy. We just fold over the list. Can we get rid of the list, and just 
fold over the `ProjectF` directly? If we had `FoldableK` we could.

What about `sets`? `Fragments.setOpt` takes a vararg `Option[Fragment]`, so we 
probably need `FoldableK` here too, but before that, how do we get the fragments? 
We probably want our `ProjectF` to store functions from the used type to `Fragment`. 
Something like `ProjectF[λ[A => (A => Fragment)]]` (I've placed parenthesis 
around the type to make it easier to read). Once we have the `Option[A]`, 
we can then map it with the function `A => Fragment`, to get a `Option[Const[Fragment]#λ[A]]`. 
Only one problem in that plan, doobie resists slightly against dealing with HKD, 
mostly when dealing with nullable columns. We also can't use the interpolator 
to make our lives easy.

First, we need a type to translate between doobie's handling of `Option` and 
our handling. When you have your doobie fragments, they contain a list of 
`Param.Elem` values. These are the values used in the prepared statement. 
For any `A` with a `Put` instance, we can create a `Param.Elem` 
using `Param.Elem.Arg(<outValue>, Put[A])`. Only problem is that the nullable 
columns (`Option[A]`) don't have a `Put` instance. We can get a `Param.Elem` 
instance for them using `Param.Elem.Opt`.

We've hit a roadblock. For all values `A`, we want to create a `Param.Elem`,
but to do so we require information about what `A` is. As that is information 
we're not given, we're not going to get a nice answer here. There is a 
solution though. We can take a `ProjectF[* => Param.Elem]` as a parameter. 
That gives us a way to convert all the values to `Param.Elem`. Let's wrap it 
in it's own type to make it a bit neater.

```scala
case class ElemCreator[A](mkElem: A => Param.Elem)
object ElemCreator {
  def arg[A](implicit put: Put[A]): ElemCreator[A]         = ElemCreator(Param.Elem.Arg(_, put))
  def opt[A](implicit put: Put[A]): ElemCreator[Option[A]] = ElemCreator(Param.Elem.Opt(_, put))
}
```

Great, we've gotten around that. Sometimes when dealing with HKD data you 
need to get around obstacles like that. Generally you just need to figure out 
how you can pass all the information you need so you don't need to inspect 
any types.

{% capture scala-doobie-equals %}
def createEquals[F[_[_]]](
    names: F[Const[List[String]]#λ],
    elemCreators: F[ElemCreator]
)(implicit F: ApplyK[F]): F[* => Fragment] =
  F.map2K(names, elemCreators)(new (Tuple2K[Const[List[String]]#λ, ElemCreator]#λ ~>: (* => Fragment)) {
    override def apply[Z](t: (List[String], ElemCreator[Z])): Z => Fragment = {
      val names      = t._1
      val columnName = names.last // The last name will be the column name
      val creator    = t._2
      
      (value: Z) => Fragment.const(columnName) ++ Fragment(" = ?", List(creator.mkElem(value)))
    }
  })
{% endcapture %}

{% capture dotty-doobie-equals %}
def createEquals[F[_[_]]](
    names: F[Const[List[String]]],
    elemCreators: F[ElemCreator]
)(given ApplyK[F]): F[[A] =>> (A => Fragment)] =
  names.map2K(elemCreators) { [Z] => (names: List[String], creator: ElemCreator[Z]) =>
    val columnName = names.last // The last name will be the column name

    (value: Z) => Fragment.const(columnName) ++ Fragment(" = ?", List(creator.mkElem(value)))
  }
{% endcapture %}

{% include code_blocks_code.html scala=scala-doobie-equals dotty=dotty-doobie-equals id="doobie-equals" %}

Then we just need a `ElemCreator` instance for `ProjectF`.
```scala
val elemCreator = ProjectF[ElemCreator](
  ElemCreator.arg,
  ElemCreator.arg,
  ProjectSettingsF[ElemCreator](
    ElemCreator.arg,
    ElemCreator.opt,
    ElemCreator.opt,
    ElemCreator.opt,
  )
)
```

Then we just need a function that combines the result of calling the method 
above with an `F[Option]`. Note that I've called these `equals` and not setters.
You could just as well use the same methods to create a doobie query method,
although that's up to you to do.

{% capture scala-doobie-equals-fragments %}
def createFragmentEquals[F[_[_]]](
    setters: F[* => Fragment],
    valuesToSet: F[Option]
)(implicit F: ApplyK[F]): F[Const[Fragment]#λ] =
  F.map2K(setters, valuesToSet)(
    λ[Tuple2K[* => Fragment, Option]#λ ~>: Const[Fragment]#λ](t => t._2.fold(Fragment.empty)(t._1))
  )
{% endcapture %}

{% capture dotty-doobie-equals-fragments %}
def createFragmentEquals[F[_[_]]](
    setters: F[[A] =>> (A => Fragment)],
    valuesToSet: F[Option]
)(given ApplyK[F]): F[[A] =>> (A => Fragment)] =
  setters.map2K(valuesToSet) { [Z] => (setter: Z => Fragment, value: Option[Z]) =>
    value.fold(Fragment.empty)(setter)
  }
{% endcapture %}

{% include code_blocks_code.html scala=scala-doobie-equals-fragments dotty=dotty-doobie-equals-fragments id="doobie-equals-fragments" %}

## Back to typeclasses
We've done what we can with the typeclasses we have up to this point. Time to create more.

### FoldableK
First up is FoldableK. This one lets us accumulate the values in the HKD to a 
single value, and leave the world of HKD. You often use FoldableK as the last 
step in a chain of transformations. It will let us remove the boilerplate 
from `hasAnyUpdates` and implement the last step of doobie's `sets`. Before we 
do that though, let's add a few more to our collection.

{% capture scala-foldableK %}
trait FoldableK[F[_[_]]] {

  def foldLeftK[A[_], B](fa: F[A], b: B)(f: B => A ~>: Const[B]#λ): B

  def foldMapK[A[_], B](fa: F[A])(f: A ~>: Const[B]#λ)(implicit B: Monoid[B]): B =
    foldLeftK(fa, B.empty)(b => λ[A ~>: Const[B]#λ](a => B.combine(b, f(a))))
    
  def toListK[A](fa: F[Const[A]#λ]): List[A] = 
    foldMapK(fa)(λ[Const[A]#λ ~>: Const[List[A]]#λ](List(_)))
}
{% endcapture %}

{% capture dotty-foldableK %}
trait FoldableK[F[_[_]]]

  def [A[_], B](fa: F[A]) foldLeftK(b: B)(f: B => A ~>: Const[B]): B

  def [A[_], B](fa: F[A]) foldMapK(f: A ~>: Const[B])(given B: Monoid[B]): B =
    fa.foldLeftK(B.empty)(b => [Z] => (fz: F[Z]) => B.combine(b, f(fz)))
    
  def [A](fa: F[Const[A]#λ]) toListK: List[A] = 
    fa.foldMapK([Z] => (a: A) => List(a))
{% endcapture %}

{% include code_blocks_code.html scala=scala-foldableK dotty=dotty-foldableK id="foldableK" %}

### TraverseK and DistributiveK
If `ApplyK` is the most useful typeclass for HKD, then these two take a close 
second place.

I mentioned way back that we had a `ProjectF[λ[A => Decoder.AccumulatingResult[Option[A]]]]`
and wanted a `Decoder.AccumulatingResult[ProjectF[Option]]` and said we'd 
need `TraverseK` for that. Just like `Traverse` lets us go 
from `F[G[A]]` to `G[F[A]]`, `TraverseK` lets us go 
from `F[[Z] =>> G[A[Z]]]` to `G[F[A]]`. Ok, so that's all good and such, but 
what is `DistributiveK`, and how is it related to `TraverseK`? 
`Distributive` is the dual of `Traverse`. Essentially it allows us to go the
other way. For example, while `Traverse` lets you go 
from `List[Option[Int]]` to `Option[List[Int]]`, `Distributive` lets you go 
from `Option[List[Int]]` to `List[Option[Int]]`, provided `List` forms a 
`Distributive` (it doesn't).

Let's do an example with ProjectF next, where `A[_]` in the above example 
is `Id` and `G[_]` is `List`. `TraverseK` then lets us go 
from `ProjectF[List]` to `List[ProjectF[Id]]`. `DistributiveK` goes the other way, 
from `List[ProjectF[Id]]` to `ProjectF[List]`. From that we can see that these 
two typeclasses essentially lets us drop in and out of HKD "mode", or if we're 
dealing with more than one type constructor on our HKD, return some data to 
the outside world that we no longer care about.

{% capture scala-traverseK-distributiveK %}
trait TraverseK[F[_[_]]] extends FunctorK[F] with FoldableK[F] {

  def traverseK[G[_]: Applicative, A[_], B[_]](fa: F[A])(f: A ~>: λ[Z => G[B[Z]]]): G[F[B]]

  def sequenceK[G[_]: Applicative, A[_]](fga: F[λ[Z => G[A[Z]]]]): G[F[A]] =
    traverseK(fga)(FunctionK.identity)(Applicative[G])

  override def mapK[A[_], B[_]](fa: F[A])(f: A ~>: B): F[B] =
    traverseK[Id, A, B](fa)(f)
}

trait DistributiveK[F[_[_]]] extends FunctorK[F] {

  def distributeK[G[_]: Functor, A[_], B[_]](gfa: G[F[A]])(f: F[λ[Z => G[A[Z]]]] ~>: B): F[B] =
    mapK(cosequenceK(gfa))(f)

  def cosequenceK[G[_]: Functor, A[_]](gfa: G[F[A]]): F[λ[Z => G[A[Z]]]]
}
{% endcapture %}

{% capture dotty-traverseK-distributiveK %}
trait TraverseK[F[_[_]]] extends extends FunctorK[F] with FoldableK[F]

  def [G[_]: Applicative, A[_], B[_]](fa: F[A]) traverseK(f: A ~>: ([Z] => G[B[Z]])): G[F[B]]

  def [G[_]: Applicative, A[_]](fga: F[[Z] => G[A[Z]]]) sequenceK: G[F[A]] =
    fga.traverseK(FunctionK.identity)(given Applicative[G])

  override def mapK[A[_], B[_]](fa: F[A])(f: A ~>: B): F[B] =
    fa.traverseK[Id, A, B](f)

trait DistributiveK[F[_[_]]] extends FunctorK[F]

  def [G[_]: Functor, A[_], B[_]](gfa: G[F[A]]) distributeK(f: F[[Z] => G[B[Z]]] ~>: B): F[B] =
    cosequenceK(gfa).mapK(f)

  def [G[_]: Functor, A[_]](gfa: G[F[A]]) cosequenceK: F[[Z] => G[A[Z]]]
{% endcapture %}

{% include code_blocks_code.html scala=scala-traverseK-distributiveK dotty=dotty-traverseK-distributiveK id="traverseK-distributiveK" %}

And that's it, no more typeclasses for the rest of this post. Let's implement 
them all for `ProjectF`, and remove all the boilerplate.

### Where's MonadK?
Some of you might wonder, "Where's `MonadK`? We've defined higher kined versions of 
all the other normal typeclasses, why not monad"? The answer is simple.

1. We don't need monad for what we're going to do. We don't need `DistributeK` 
either, but it's so tightly bound to `TraverseK` when dealing with HKD, that
I felt it was best to show it.
2. Unlike everything else we've gone over to far, implementing `MonadK` 
for `ProjectF` is much less obvious how to do correctly. (Yes, `ProjectF` does 
form a `MonadK`).
3. It's hard to wrap your head around. Why is it useful? Trying to 
understand `MonadK` is probably the closest you'll ever get to going back to 
before you understood `Monad`. I'm not quite there yet myself.

I'll probably cover `MonadK` at some later point in it's own post.

## Putting it all together.
Not that we have all the typeclasses we'll need, let's put the to use. First 
stop, implementing them for `ProjectF` and `ProjectSettingsF`.

{% capture scala-projectF-all-instances %}
val F: ApplicativeK[ProjectF] with TraverseK[ProjectF] with DistributeK[ProjectF] =
  new ApplicativeK[ProjectF] with TraverseK[ProjectF] with DistributeK[ProjectF] {
  
    def pureK[A[_]](a: Const[Unit]#λ ~>: A): ProjectF[A] = ProjectF[A](
      a(()),
      a(()),
      implicitly[ApplicativeK[ProjectSettingsF]].pureK(a)
    )
    
    def map2K[A[_], B[_], Z[_]](fa: ProjectF[A], fb: ProjectF[B])(f: Tuple2K[A, B]#λ ~>: Z): ProjectF[Z] = 
      ProjectF[Z](
        f((fa.name, fb.name)),
        f((fa.description, fb.description)),
        implicitly[ApplyK[ProjectSettingsF]].map2K(fa.settings, fb.settings)(f),
      )
      
    def foldLeftK[A[_], B](fa: F[A], b: B)(f: B => A ~>: Const[B]#λ): B = {
      val b1 = f(b)(fa.name)
      val b2 = f(b1)(fa.description)
      implicitly[FoldableK[ProjectSettingsF]].foldLeftK(fa.settings, b2)(f)
    }
    
    def traverseK[G[_]: Applicative, A[_], B[_]](fa: ProjectF[A])(f: A ~>: λ[Z => G[B[Z]]]): G[ProjectF[B]] = 
      (
        fa.name, 
        fa.description, 
        implicitly[TraverseK[ProjectSettingsF]].traverseK(fa.settings)(f)
      ).mapN(ProjectF.apply)
    
    def cosequenceK[G[_]: Functor, A[_]](gfa: G[ProjectF[A]]): ProjectF[λ[Z => G[A[Z]]]] = 
      ProjectF[λ[Z => G[A[Z]]]](
        gfa.map(_.name),
        gfa.map(_.description),
        implicitly[DistributeK[ProjectSettingsF]].cosequenceK(gfa.map(_.settings))
      )
  }

val F: ApplicativeK[ProjectSettingsF] with TraverseK[ProjectSettingsF] with DistributeK[ProjectSettingsF] =
  new ApplicativeK[ProjectSettingsF] with TraverseK[ProjectSettingsF] with DistributeK[ProjectSettingsF] {
  
    def pureK[A[_]](a: Const[Unit]#λ ~>: A): ProjectSettingsF[A] = ProjectSettingsF[A](
      a(()),
      a(()),
      a(()),
      a(())
    )
    
    def map2K[A[_], B[_], Z[_]](fa: ProjectSettingsF[A], fb: ProjectSettingsF[B])(f: Tuple2K[A, B]#λ ~>: Z): ProjectSettingsF[Z] = 
      ProjectSettingsF[Z](
        f((fa.keywords, fb.keywords)),
        f((fa.issues, fb.issues)),
        f((fa.sources, fb.sources)),
        f((fa.supportChannel, fb.supportChannel))
      )
    
    def foldLeftK[A[_], B](fa: F[A], b: B)(f: B => A ~>: Const[B]#λ): B = {
      val b1 = f(b)(fa.keywords)
      val b2 = f(b1)(fa.issues)
      val b3 = f(b2)(fa.sources)
      f(b3)(fa.supportChannel)
    }
    
    def traverseK[G[_]: Applicative, A[_], B[_]](fa: ProjectSettingsF[A])(f: A ~>: λ[Z => G[B[Z]]]): G[ProjectSettingsF[B]] = 
      (
        fa.keywords, 
        fa.issues,
        fa.sources,
        fa.supportChannel,
      ).mapN(ProjectSettingsF.apply)
    
    def cosequenceK[G[_]: Functor, A[_]](gfa: G[ProjectSettingsF[A]]): ProjectSettingsF[λ[Z => G[A[Z]]]] = 
      ProjectSettingsF[λ[Z => G[A[Z]]]](
        gfa.map(_.keywords),
        gfa.map(_.issues),
        gfa.map(_.sources),
        gfa.map(_.supportChannel),
      )
  }
{% endcapture %}

{% capture dotty-projectF-all-instances %}
given ApplicativeK[ProjectF], TraverseK[ProjectF], DistributeK[ProjectF]
  
  def [A[_]](a: ValueK[A]) pureK: ProjectF[A] = ProjectF(
    a(),
    a(),
    a.pureK
  )
  
  def [A[_], B[_], Z[_]](fa: ProjectF[A]) map2K(fb: ProjectF[B])(f: Function2K[A, B, Z]): ProjectF[Z] = 
    ProjectF(
      f(fa.name, fb.name),
      f(fa.description, fb.description),
      fa.settings.map2K(fb.settings)(f)
    )

  def [A[_], B](fa: F[A]) foldLeftK(b: B)(f: B => A ~>: Const[B]): B = 
    val b1 = f(b)(fa.name)
    val b2 = f(b1)(fa.description)
    fa.settings.foldLeftK(b2)(f)
  
  def [G[_]: Applicative, A[_], B[_]](fa: ProjectF[A]) traverseK(f: A ~>: ([Z] => G[B[Z]])): G[ProjectF[B]] = 
    (fa.name, fa.description, fa.settings.traverseK(f)).mapN(ProjectF.apply)
  
  def [G[_]: Functor, A[_]](gfa: G[ProjectF[A]]) cosequenceK: ProjectF[[Z] => G[A[Z]]] = 
    ProjectF(
      gfa.map(_.name),
      gfa.map(_.description),
      gfa.map(_.settings).cosequenceK
    )

given ApplicativeK[ProjectSettingsF], TraverseK[ProjectSettingsF], DistributeK[ProjectSettingsF]
  
  def [A[_]](a: ValueK[A]) pureK: ProjectSettingsF[A] = ProjectSettingsF(
    a(),
    a(),
    a(),
    a()
  )
  
  def [A[_], B[_], Z[_]](fa: ProjectSettingsF[A]) map2K(fb: ProjectSettingsF[B])(f: Function2K[A, B, Z]): ProjectSettingsF[Z] = 
    ProjectSettingsF(
      f(fa.keywords, fb.keywords),
      f(fa.issues, fb.issues),
      f(fa.sources, fb.sources),
      f(fa.supportChannel, fb.supportChannel)
    )

  def [A[_], B](fa: F[A]) foldLeftK(b: B)(f: B => A ~>: Const[B]): B = 
    val b1 = f(b)(fa.keywords)
    val b2 = f(b1)(fa.issues)
    val b2 = f(b2)(fa.sources)
    f(b3)(fa.supportChannel)
  
  def [G[_]: Applicative, A[_], B[_]](fa: ProjectSettingsF[A]) traverseK(f: A ~>: ([Z] => G[B[Z]])): G[ProjectSettingsF[B]] = 
    (fa.keywords, fa.issues, fa.sources, fa.supportChannel).mapN(ProjectSettingsF.apply)
  
  def [G[_]: Functor, A[_]](gfa: G[ProjectSettingsF[A]]) cosequenceK: ProjectSettingsF[[Z] => G[A[Z]]] = 
    ProjectSettingsF(
      gfa.map(_.keywords),
      gfa.map(_.issues),
      gfa.map(_.sources),
      gfa.map(_.supportChannel)
    )
{% endcapture %}

{% include code_blocks_code.html scala=scala-projectF-all-instances dotty=dotty-projectF-all-instances id="projectF-all-instances" %}

As you can see this is a lot of code, and a wonderful place to let a macro do 
the job for you. Could we delegate this job to shapeless? So far we've been
avoiding it, but this looks like a good fit for it. Deriving low level code.
In theory yes, but in practice no. Shapeless does not support higher kinded data,
so we could only derive stuff specialized to some type we're interested in.

### Fixing patchDecode
Now that we have `TraverseK`, we can finally fix `patchDecode` and give it the 
right type. We just need to insert a `sequenceK`

{% capture scala-patch-decode %}
def patchDecode[F[_[_]]](names: F[Const[List[String]]#λ], decoders: F[Decoder], cursor: ACursor)(
    implicit F: ApplyK[F],
    FT: TraverseK[F]
): Decoder.AccumulatingResult[F[Option]] = {
  val hkdRes = F.map2K(names, decoders)(
    new (Tuple2K[Const[List[String]]#λ, Decoder]#λ ~>: λ[A => Decoder.AccumulatingResult[Option[A]]]) {
      override def apply[Z](t: (List[String], Decoder[Z])): Decoder.AccumulatingResult[Option[Z]] = {
        val names   = t._1
        val decoder = t._2

        val cursorWithNames = names.foldLeft(cursor)(_.downField(_))

        val result = 
          if (cursorWithNames.succeeded) Some(decoder.decode(cursorWithNames)) 
          else None
        result.sequence.toValidatedNel
      }
    }
  )
  
  FT.sequenceK(hkdRes)
}
{% endcapture %}

{% capture dotty-patch-decode %}
def patchDecode[F[_[_]]](names: F[Const[List[String]]], decoders: F[Decoder], cursor: ACursor)(
    given ApplyK[F], TraverseK[F]
): Decoder.AccumulatingResult[F[Option]] =
  names.map2K(decoders) { [Z] => (names: List[String], decoders: Decoder[Z]) =>
    val cursorWithNames = names.foldLeft(cursor)(_.downField(_))

    val result = 
      if cursorWithNames.succeeded then Some(decoder.decode(cursorWithNames)) 
      else None
    result.sequence.toValidatedNel
  }.sequenceK
{% endcapture %}

{% include code_blocks_code.html scala=scala-patch-decode dotty=dotty-patch-decode id="patch-decode" %}

### Folding hasAnyUpdates
Next up is the `hasAnyUpdates` check. Here we just need to fold over the 
structure, and see if any fields are defined.

{% capture scala-has-any-updates-fold %}
def hasAnyUpdates[F[_[_]]](fields: F[Option])(implicit F: FoldableK[F]): Boolean = 
  F.foldLeftK(fields, false)(b => λ[Option ~>: Const[Boolean]#λ](opt => b || opt.isDefined))
{% endcapture %}

{% capture dotty-has-any-updates-fold %}
def hasAnyUpdates[F[_[_]]](fields: F[Option])(given FoldableK[F]): Boolean = 
  fields.foldLeftK(false)(b => [A] => (optA: Option[A]) => b || optA.isDefined)
{% endcapture %}

{% include code_blocks_code.html scala=scala-has-any-updates-fold dotty=dotty-has-any-updates-fold id="has-any-updates-fold" %}

### doobie setter
Next, let's finish our doobie updater. Let's build off `createFragmentEquals` 
which gives us `F[Const[Fragment]#λ]` provided we pass in the setters, and the 
values we want to update.

{% capture scala-doobie-setter %}
def updateTable[F[_[_]], A: Put](
    setters: F[* => Fragment],
    tableName: String,
    identifierColumn: String
)(values: F[Option], identifier: A)(implicit F: ApplyK[F], FF: FoldableK[F]): ConnectionIO[Int] = {
  val sets = Fragments.set(FF.toListK(createFragmentEquals(setters, valuesToSet)): _*)
  val cond = Fragment.const(identifierColumn) ++ fr"= $identifier"
  
  val query = sql"UPDATE " ++ Fragment.const(tableName) ++ sets ++ cond
  query.update.run
}
{% endcapture %}

{% capture dotty-doobie-setter %}
def updateTable[F[_[_]], A: Put](
    setters: F[[A] => (A => Fragment)],
    tableName: String,
    identifierColumn: String
)(values: F[Option], identifier: A)(given ApplyK[F], FoldableK[F]): ConnectionIO[Int] =
  val sets = Fragments.set(createFragmentEquals(setters, valuesToSet).toListK: _*)
  val cond = Fragment.const(identifierColumn) ++ fr"= $identifier"
  
  val query = sql"UPDATE " ++ Fragment.const(tableName) ++ sets ++ fr"WHERE" ++ cond
  query.update.run
{% endcapture %}

{% include code_blocks_code.html scala=scala-doobie-setter dotty=dotty-doobie-setter id="doobie-setter" %}

### Wrapping it all up
And to tie it all together, let's define a real version of `handlePatch` that we can call with `ProjectF`.

{% capture scala-handle-patch %}
def handlePatch[F[_[_]], A: Put](
    names: F[Const[List[String]]#λ],
    decoders: F[Decoder],
    tableName: String, 
    identifierColumn: String, 
    identifier: String, 
    json: Json,
    identifier: A
)(implicit F: ApplyK[F], FT: TraverseK[F]): IO[Result] = {
  val partialThingResult = patchDecode(names, decoders, json.hcursor)
  
  partialThingResult.fold(handleDecodeError) { partialThing =>
    val update = updateTable(setters, tableName, identifierColumn)(partialThing, identifier)
  
    if (hasAnyUpdates(partialThing)) dbService.run(update).map(_ => NoContent)
    else IO.pure(BadRequest("No updates specified"))
  }
}
{% endcapture %}

{% capture dotty-handle-patch %}
def handlePatch[F[_[_]], A: Put](
    names: F[Const[List[String]]],
    decoders: F[Decoder],
    tableName: String, 
    identifierColumn: String, 
    identifier: String, 
    json: Json,
    identifier: A
)(given ApplyK[F], TraverseK[F]): IO[Result] =
  val partialThingResult = patchDecode(names, decoders, json.hcursor)
  
  partialThingResult.fold(handleDecodeError) { partialThing =>
    val update = updateTable(setters, tableName, identifierColumn)(partialThing, identifier)
  
    if hasAnyUpdates(partialThing) then dbService.run(update).map(_ => NoContent)
    else IO.pure(BadRequest("No updates specified"))
  }
{% endcapture %}

{% include code_blocks_code.html scala=scala-handle-patch dotty=dotty-handle-patch id="handle-patch" %}

We finally did it. We now have a generic path method that perfectly models 
our intent. Not one place did we have to resort to logic programming. We 
gained more than just `handlePatch`. Best example is to look at `updateTable`.
Here we're using it to update the DB based on the passed in JSON, but nothing
says we're restricted to just that. Let's specialize it for `ProjectF` to get 
a better idea of what we have.

```scala
def updateProject(values: ProjectF[Option], projectId: String) = 
  updateTable(ProjectF.elemCreator, "projects", "project_id")(values, projectId)
```
Turns out we can also use `updateTable` to do arbitrary simple updates to our
models. It's not the only method we can potentially use in other places. 
If you look at `patchDecode` it doesn't look too hard to turn that into a 
typeclass. Can we then use something similar to derive JSON decoders for 
case classes? Yes, turns out that HKD is very good at modeling typeclass 
derivation, both for HKD and non-HKD, but more about that in some other post.

## Closing words
I hope this post has given you some ideas of what HKD is, and more importantly 
how it can be used. Rest assured that we've just been scratching the surface 
of HKD in this post. There is a lot more left to cover. Here's a rough plan on 
what I have planned so far.

### Future plans
* **Nesting:** You might have noticed in some places that even though we 
defined our structure as two classes `ProjectF` and `ProjectSettingsF`, it 
didn't act like it, as we're used to. Why? How can we use this? And most 
interestingly, what happes when we stick something other than `F[A]` into a 
HKD type? For example, is `List[F[Int]]` valid? We'll also take a look a 
sum HKD here.
* **HKD Typeclass derivation:** In this post we saw many places where we could
have used typeclass derivation to further simplify the code a bit. I didn't do
that here to keep stuff simple. Next up after we've talked about nesting is
how to derive typeclasses for HKD, and what building blocks we'll need. (
You've already seen `Const[List[String]]`, which is one such building block, 
but there are others).
* **ADT Typeclass derivation:** While HKD is very useful, and gives us lots of
stuff for free, sometimes we just want to define normal ADTs instead. That 
doesn't mean that we have to give up the power HKD give to us however. 
Just like Shapeless 3 will add some of the operations we've seen here, like 
`mapK`, `map2K`, `foldK` and such, we can take some building blocks from 
Shapeless to allow us to derive typeclasses for simple ADTs.
* **MonadK:** Monad tutorial incoming eventually I guess. At that point we
can talk about `MonadK`, what it is, how it works, and what it can be used for.
We'll also look at another typeclass, more powerful than anything we've seen 
so far. In exchange for such power, we must pay a price, the higher the price, 
the more powerful the typeclass becomes.

### Playing around with stuff
If you want to play around with HKD a bit more, I have extracted a lot of my 
HKD code into a library called Perspective. You can find it here (TODO).
It contains the basic typeclasses we've talked about here, in addition to
some building blocks for future topics. It also contains code to derive the
typeclasses we've defined here (`FunctorK`, `ApplyK`,`ApplicativeK`, 
`FoldableK`, `TraverseK`, `DistributeK`). Do note however that I've simplified
things a bit for this blog post, and that Perspective holds back a bit less.
